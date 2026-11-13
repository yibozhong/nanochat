"""Bounded real-shape baseline/feedback comparison; local JSON only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import torch
from nanochat.connection_cost import model_cost
from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
from nanochat.packed_training import PackedData, evaluate, json_write, seed_all, train_step


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',required=True)
    p.add_argument('--output-dir',required=True)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    configure_attention('fa2');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    data=PackedData(a.data_dir);models={};timings={};losses={}
    assert len(data.arrays['train'])>=96*16
    for label,feedback in [('baseline',False),('feedback',True)]:
        seed_all(42)
        model=build_connection_model(ConnectionConfig(streams=1,latent_feedback=feedback)).prepare(compile_blocks=True)
        opt,wd=optimizer_for(model)
        models[label]=(model,opt,wd);timings[label]=[];losses[label]=[]
    baseline=models['baseline'][0];feedback=models['feedback'][0]
    assert all(torch.equal(v,feedback.state_dict()[k]) for k,v in baseline.state_dict().items())
    torch.cuda.reset_peak_memory_stats()
    with (out/'steps.jsonl').open('w',buffering=1) as log:
        for step in range(96):
            # Alternate order to reduce warmup/clock-drift bias on one GPU.
            for label in (('baseline','feedback') if step%2==0 else ('feedback','baseline')):
                m,opt,wd=models[label];schedule(opt,step,305176,wd)
                torch.cuda.synchronize();start=time.perf_counter()
                loss=train_step(m,opt,data,step,microbatch=16,seed=42)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                passes=m.training_passes(42,step) if m.config.latent_feedback else 1
                row=dict(model=label,step=step+1,loss=loss,seconds=elapsed,passes=passes)
                log.write(json.dumps(row)+'\n')
                losses[label].append(loss)
                if step>=16:timings[label].append((passes,elapsed))
                if step<3 or (step+1)%16==0:print(json.dumps(row),flush=True)
    result=dict(complete=True,formal_training=False,gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,attention_backend='fa2',
        seed=42,steps_per_model=96,tokens_per_model=96*32768,microbatch=16,
        data_manifest=data.manifest,models={},
        combined_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        memory_note='Both models and optimizers resident for alternating same-GPU timing.')
    for label,(m,opt,wd) in models.items():
        info=dict(config=asdict(m.config),parameters=sum(p.numel() for p in m.parameters()),
            first_loss=losses[label][0],last_loss=losses[label][-1],
            mean_step_seconds=statistics.mean(t for _,t in timings[label]),
            median_step_seconds=statistics.median(t for _,t in timings[label]),
            cost=model_cost(m.config,batch=16))
        if m.config.latent_feedback:
            info['dev_ce_p1']=evaluate(m,data,passes=1)[0]
            info['dev_ce_p4']=evaluate(m,data,passes=4)[0]
            info['measured_pass_counts']={str(p):sum(q==p for q,_ in timings[label]) for p in (1,2)}
            info['mean_seconds_by_pass']={str(p):statistics.mean(t for q,t in timings[label] if q==p) for p in (1,2)}
        else:info['dev_ce_p1']=evaluate(m,data)[0]
        result['models'][label]=info
    result['measured_training_time_ratio']=result['models']['feedback']['mean_step_seconds']/result['models']['baseline']['mean_step_seconds']
    result['expected_training_flops_ratio']=model_cost(feedback.config)['estimated_training_flops']/model_cost(baseline.config)['estimated_training_flops']
    json_write(out/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('models','data_manifest')}),flush=True)


if __name__=='__main__':main()
