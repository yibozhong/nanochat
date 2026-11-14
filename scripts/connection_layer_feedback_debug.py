"""Same-GPU full-shape timing and bounded training before the authorized run."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import time

import torch
import triton

from nanochat.connection_cost import model_cost
from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig,build_connection_model,optimizer_for,schedule
from nanochat.gpt import norm
from nanochat.layer_feedback import layer_read
from nanochat.packed_training import PackedData,evaluate,json_write,seed_all,train_step


def operator_benchmark():
    seed_all(91)
    q=torch.randn(16,2048,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    bank=torch.randn(16,2048,8,512,device='cuda',dtype=torch.bfloat16)
    grad=torch.randn_like(q)
    results={}
    for backend in ('torch','triton'):
        fn=torch.compile(lambda x,b: layer_read(x,b,backend=backend),fullgraph=True,dynamic=False)
        def step():
            out=fn(q,bank)
            return torch.autograd.grad(out,q,grad)[0]
        for _ in range(5):step()
        results[backend+'_forward_backward_ms']=triton.testing.do_bench(step,warmup=100,rep=300)
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    configure_attention('fa2');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    operator=operator_benchmark();gc.collect();torch.cuda.empty_cache()
    data=PackedData(a.data_dir);models={};times={};losses={}
    settings=[('baseline',False,'top','triton'),('top',True,'top','triton'),
        ('layers_reference',True,'layers','torch'),('layers_triton',True,'layers','triton')]
    for label,feedback,source,backend in settings:
        seed_all(42)
        c=ConnectionConfig(streams=1,latent_feedback=feedback,feedback_source=source,feedback_attention_backend=backend)
        model=build_connection_model(c).prepare(compile_blocks=True)
        opt,wd=optimizer_for(model)
        models[label]=(model,opt,wd);times[label]=[];losses[label]=[]
    backbone=models['baseline'][0].state_dict()
    assert all(torch.equal(p,m.state_dict()[n]) for m,_,_ in models.values() for n,p in backbone.items())
    query_before=models['layers_triton'][0].feedback_query.weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    labels=list(models)
    with (out/'steps.jsonl').open('w',buffering=1) as log:
        for step in range(96):
            order=labels[step%4:]+labels[:step%4]
            for label in order:
                model,opt,wd=models[label];schedule(opt,step,305176,wd)
                torch.cuda.synchronize();start=time.perf_counter()
                loss=train_step(model,opt,data,step,microbatch=16,seed=42)
                torch.cuda.synchronize();duration=time.perf_counter()-start
                passes=model.training_passes(42,step) if model.config.latent_feedback else 1
                row=dict(model=label,step=step+1,loss=loss,seconds=duration,passes=passes)
                losses[label].append(loss)
                if step>=16:times[label].append((passes,duration))
                log.write(json.dumps(row)+'\n')
                if step<2 or (step+1)%16==0:print(json.dumps(row),flush=True)
    result=dict(complete=True,formal_training=False,gpu=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),torch=torch.__version__,cuda=torch.version.cuda,
        attention_backend='fa2',steps_per_model=96,tokens_per_model=96*32768,batch_tokens=32768,
        microbatch=16,seed=42,data_manifest=data.manifest,operator=operator,models={},
        combined_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        memory_note='Four models/optimizers resident for rotating-order same-GPU timing.')
    for label,(model,opt,wd) in models.items():
        info=dict(config=asdict(model.config),parameters=sum(p.numel() for p in model.parameters()),
            first_loss=losses[label][0],last_loss=losses[label][-1],
            mean_step_seconds=statistics.mean(t for _,t in times[label]),
            median_step_seconds=statistics.median(t for _,t in times[label]),
            cost=model_cost(model.config,batch=16))
        if model.config.latent_feedback:
            info['dev_ce_p1']=evaluate(model,data,passes=1)[0]
            info['dev_ce_p4']=evaluate(model,data,passes=4)[0]
            info['mean_seconds_by_pass']={str(p):statistics.mean(t for n,t in times[label] if n==p) for p in (1,2)}
        else:info['dev_ce_p1']=evaluate(model,data)[0]
        result['models'][label]=info
    model=models['layers_triton'][0]
    change=(model.feedback_query.weight.detach()-query_before).abs().mean().item()
    assert change>1e-8,'Query projection did not learn'
    with torch.no_grad():
        x,_=data.batch('dev',0,1)
        embedding=norm(model.wte(x));_,bank=model.backbone(embedding,return_layers=True)
        bank=bank[:,:-1].float();bank=bank*torch.rsqrt(bank.square().mean(-1,keepdim=True)+1e-5)
        query=model.feedback_query(embedding[:,1:]).float()
        probs=((query.unsqueeze(2)*bank).sum(-1)*(model.config.width**-.5)).softmax(-1)
        result['layer_selection']=dict(query_mean_abs_change=change,
            mean_weight_by_layer=probs.mean((0,1)).tolist(),
            token_std_by_layer=probs.std((0,1)).tolist(),
            mean_entropy=(-(probs*probs.clamp_min(1e-20).log()).sum(-1)).mean().item())
    latency=result['models']['layers_triton']['mean_step_seconds']
    result['ratio_to_baseline']=latency/result['models']['baseline']['mean_step_seconds']
    result['ratio_to_top']=latency/result['models']['top']['mean_step_seconds']
    result['ratio_to_reference']=latency/result['models']['layers_reference']['mean_step_seconds']
    json_write(out/'summary.json',result)
    # Stop for investigation before formal training if the optimized path regresses heavily.
    assert result['ratio_to_top']<1.35,result['ratio_to_top']
    assert result['ratio_to_reference']<1.20,result['ratio_to_reference']
    print(json.dumps({k:v for k,v in result.items() if k not in ('models','data_manifest')}),flush=True)


if __name__=='__main__':main()
