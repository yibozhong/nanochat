"""Bounded production-shape correctness, timing and memory; no W&B run."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import time
from contextlib import nullcontext

import torch

from nanochat.connection_cost import model_cost
from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig,build_connection_model,optimizer_for,schedule
from nanochat.packed_training import PackedData,evaluate,json_write,seed_all,train_step
from scripts.connection_feedback_efficiency import component_benchmark,profile_step


def profile(source,data,out,order='reassociated',steps=96):
    seed_all(42)
    c=ConnectionConfig(streams=1,latent_feedback=source!='baseline',
        feedback_source='top' if source=='baseline' else source,feedback_projection_order=order)
    label=source+('_materialized' if order=='materialized' else '')
    model=build_connection_model(c).prepare(compile_blocks=True)
    opt,wd=optimizer_for(model)
    initial={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('feedback_readers.')}
    before_bytes=torch.cuda.memory_allocated()
    peak_bytes=0;peak_reserved=0;traced=False
    times=[];losses=[]
    with (out/f'{label}-steps.jsonl').open('w',buffering=1) as log:
        for step in range(steps):
            schedule(opt,step,305176,wd)
            passes=model.training_passes(42,step) if c.latent_feedback else 1
            instrument=not traced and step>=steps-16 and (passes==2 or not c.latent_feedback)
            context=profile_step(model,opt,out/f'{label}-cuda') if instrument else nullcontext()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize();start=time.perf_counter()
            with context:loss=train_step(model,opt,data,step,microbatch=16,seed=42)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            row=dict(model=label,step=step+1,loss=loss,seconds=seconds,passes=passes,profiled=instrument)
            losses.append(loss)
            if instrument:traced=True
            else:
                peak_bytes=max(peak_bytes,torch.cuda.max_memory_allocated())
                peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved())
                if step>=16:times.append((passes,seconds))
            log.write(json.dumps(row)+'\n')
            if step<2 or (step+1)%16==0:print(json.dumps(row),flush=True)
    result=dict(config=asdict(c),parameters=sum(p.numel() for p in model.parameters()),
        first_loss=losses[0],last_loss=losses[-1],mean_step_seconds=statistics.mean(t for _,t in times),
        median_step_seconds=statistics.median(t for _,t in times),
        mean_seconds_by_pass={str(p):statistics.mean(t for n,t in times if n==p) for p in sorted({n for n,_ in times})},
        allocated_before_training_bytes=before_bytes,peak_allocated_bytes=peak_bytes,
        peak_reserved_bytes=peak_reserved,cost=model_cost(c,batch=16),steps=steps,tokens=steps*32768,
        cuda_profile=f'{label}-cuda.json',cuda_trace=f'{label}-cuda.trace.json.gz')
    change={n:(p.detach()-initial[n]).abs().mean().item() for n,p in model.named_parameters() if n in initial}
    assert all(v>1e-8 for v in change.values()),change
    result['projection_mean_abs_changes']=change
    if c.latent_feedback:
        result['dev_ce_p1']=evaluate(model,data,passes=1)[0]
        result['dev_ce_p4']=evaluate(model,data,passes=4)[0]
    else:result['dev_ce_p1']=evaluate(model,data)[0]
    assert traced,'Missing warmed CUDA profile'
    json_write(out/f'{label}.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    configure_attention('fa2');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    operator=component_benchmark();json_write(out/'components.json',operator)
    gc.collect();torch.cuda.empty_cache()
    data=PackedData(a.data_dir)
    result=dict(complete=False,formal_training=False,gpu=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),torch=torch.__version__,cuda=torch.version.cuda,
        attention_backend='fa2',steps_per_model=96,tokens_per_model=96*32768,batch_tokens=32768,
        microbatch=16,seed=42,data_manifest=data.manifest,operator=operator,models={},
        timing_note='Same GPU, first 16 updates and instrumented update excluded. One model resident at a time. Training peak excludes instrumented step. Materialized comparisons use 48 steps; four main models use 96.')
    settings=[(s,'reassociated',96) for s in ('baseline','layers','embedding_qkv','per_layer_qkv')]
    settings += [(s,'materialized',48) for s in ('embedding_qkv','per_layer_qkv')]
    for source,order,steps in settings:
        label=source+('_materialized' if order=='materialized' else '')
        result['models'][label]=profile(source,data,out,order,steps)
        gc.collect();torch.cuda.empty_cache()
        json_write(out/'summary.json',result)
    old=result['models']['layers']['mean_step_seconds']
    for name in ('embedding_qkv','per_layer_qkv'):
        result['models'][name]['ratio_to_unprojected_feedback']=result['models'][name]['mean_step_seconds']/old
        # Compare P2 directly: the sampled P1 fraction differs in 48/96 steps.
        fast=result['models'][name];slow=result['models'][name+'_materialized']
        fast['p2_speedup_from_reassociation']=slow['mean_seconds_by_pass']['2']/fast['mean_seconds_by_pass']['2']
        fast['peak_bytes_saved']=slow['peak_allocated_bytes']-fast['peak_allocated_bytes']
    result['complete']=True;json_write(out/'summary.json',result)
    print('PROJECTED_FEEDBACK_PROFILE_COMPLETE',flush=True)


if __name__=='__main__':main()
