"""Bounded production-shape correctness, timing and memory; no W&B run."""
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
from nanochat.packed_training import PackedData,evaluate,json_write,seed_all,train_step
from nanochat.projected_layer_read import projected_read


def operator_benchmark():
    seed_all(91)
    q=torch.randn(16,2048,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    kv=torch.randn(16,2048,8,1024,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    grad=torch.randn_like(q);result={}
    for backend in ('torch','triton'):
        fn=torch.compile(lambda x,b: projected_read(x,b,backend=backend),fullgraph=True,dynamic=False)
        def step():
            out=fn(q,kv)
            return torch.autograd.grad(out,(q,kv),grad)
        for _ in range(5):step()
        result[backend+'_forward_backward_ms']=triton.testing.do_bench(step,warmup=100,rep=300)
    return result


def profile(source,data,out):
    seed_all(42)
    c=ConnectionConfig(streams=1,latent_feedback=source!='baseline',
        feedback_source='top' if source=='baseline' else source)
    model=build_connection_model(c).prepare(compile_blocks=True)
    opt,wd=optimizer_for(model)
    initial={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('feedback_readers.')}
    before_bytes=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    times=[];losses=[]
    with (out/f'{source}-steps.jsonl').open('w',buffering=1) as log:
        for step in range(96):
            schedule(opt,step,305176,wd)
            torch.cuda.synchronize();start=time.perf_counter()
            loss=train_step(model,opt,data,step,microbatch=16,seed=42)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            passes=model.training_passes(42,step) if c.latent_feedback else 1
            row=dict(model=source,step=step+1,loss=loss,seconds=seconds,passes=passes)
            losses.append(loss)
            if step>=16:times.append((passes,seconds))
            log.write(json.dumps(row)+'\n')
            if step<2 or (step+1)%16==0:print(json.dumps(row),flush=True)
    result=dict(config=asdict(c),parameters=sum(p.numel() for p in model.parameters()),
        first_loss=losses[0],last_loss=losses[-1],mean_step_seconds=statistics.mean(t for _,t in times),
        median_step_seconds=statistics.median(t for _,t in times),
        mean_seconds_by_pass={str(p):statistics.mean(t for n,t in times if n==p) for p in sorted({n for n,_ in times})},
        allocated_before_training_bytes=before_bytes,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),cost=model_cost(c,batch=16))
    change={n:(p.detach()-initial[n]).abs().mean().item() for n,p in model.named_parameters() if n in initial}
    assert all(v>1e-8 for v in change.values()),change
    result['projection_mean_abs_changes']=change
    if c.latent_feedback:
        result['dev_ce_p1']=evaluate(model,data,passes=1)[0]
        result['dev_ce_p4']=evaluate(model,data,passes=4)[0]
    else:result['dev_ce_p1']=evaluate(model,data)[0]
    json_write(out/f'{source}.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    configure_attention('fa2');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    operator=operator_benchmark();gc.collect();torch.cuda.empty_cache()
    data=PackedData(a.data_dir)
    result=dict(complete=False,formal_training=False,gpu=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),torch=torch.__version__,cuda=torch.version.cuda,
        attention_backend='fa2',steps_per_model=96,tokens_per_model=96*32768,batch_tokens=32768,
        microbatch=16,seed=42,data_manifest=data.manifest,operator=operator,models={},
        timing_note='Sequential runs on the same GPU; first 16 updates excluded. One model resident at a time; measured training peak includes optimizer and saved activations.')
    for source in ('baseline','layers','embedding_qkv','per_layer_qkv'):
        result['models'][source]=profile(source,data,out)
        gc.collect();torch.cuda.empty_cache()
        json_write(out/'summary.json',result)
    old=result['models']['layers']['mean_step_seconds']
    for name in ('embedding_qkv','per_layer_qkv'):
        result['models'][name]['ratio_to_unprojected_feedback']=result['models'][name]['mean_step_seconds']/old
    result['complete']=True;json_write(out/'summary.json',result)
    print('PROJECTED_FEEDBACK_PROFILE_COMPLETE',flush=True)


if __name__=='__main__':main()
