"""Bounded single-GPU diagnosis; never changes the production training path."""
import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import gzip
import gc
import json
import os
from pathlib import Path
import statistics
import time
import uuid

import torch
from torch.profiler import profile, ProfilerActivity, record_function
import wandb

from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.packed_training import PackedData, seed_all, train_step, json_write


def summarize(samples):
    return dict(samples=samples, mean=statistics.mean(samples), median=statistics.median(samples),
                stdev=statistics.stdev(samples) if len(samples)>1 else 0.)


def phased_step(model, opt, data, step, *, finite_check=True, labels=False, events=False):
    """Same micro16/accum1 operations as packed_training.train_step, with ranges.

    CUDA event intervals include stream gaps; they are not pure kernel times.
    No intermediate synchronization is added between phases.
    """
    stamps={};cpu={}
    @contextmanager
    def phase(name):
        begin=end=None
        if events:
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record()
        started=time.perf_counter()
        with record_function('phase/'+name) if labels else nullcontext():
            yield
        cpu[name]=(time.perf_counter()-started)*1000
        if events:end.record();stamps[name]=(begin,end)
    with phase('zero_grad'):
        opt.zero_grad(set_to_none=True)
        total=torch.zeros((),device='cuda')
    with phase('data'):
        x,y=data.batch('train',step*16,16)
    with phase('forward'):
        loss=model(x,y)
    with phase('backward'):
        (loss/1).backward()
        total+=loss.detach()/1
    with phase('finite_check'):
        if finite_check:
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            finite=torch.stack([torch.isfinite(g).all() for g in grads]).all()
            if not bool(finite) or not torch.isfinite(total):
                raise FloatingPointError('Nonfinite gradient/loss in profiling')
    with phase('optimizer'):
        opt.step()
    with phase('loss_item'):
        value=total.item()
    if events:
        torch.cuda.synchronize()
        return value,dict(cpu_wall_ms=cpu,stream_elapsed_ms={k:a.elapsed_time(b) for k,(a,b) in stamps.items()})
    return value


def kernel_category(name):
    if 'mhc_pre_norm_fn_bwd_mul' in name:return 'connection_projection_backward_gemm'
    if 'mhc_pre_norm_fn_fwd_mul' in name:return 'connection_projection_forward_gemm'
    if 'mhc_pre_norm_fn' in name:return 'connection_projection_norm'
    if 'sinkhorn' in name.lower():return 'sinkhorn'
    if 'mhc_pre_apply_mix' in name:return 'connection_read'
    if 'mhc_post' in name:return 'connection_write'
    if 'mhc_pre_split_mixes' in name:return 'connection_coefficients'
    if 'mhc_head' in name:return 'connection_head_coefficients'
    if 'expand_to_mhc' in name:return 'connection_expand'
    if any(x in name.lower() for x in ('flash','fmha','attention','sdpa')):return 'attention'
    if 'cce' in name.lower() or 'cross_entropy' in name.lower():return 'cross_entropy'
    return 'other_backbone_optimizer_or_framework'


def trace_summary(path, steps):
    trace=json.loads(path.read_text())
    kernels=[e for e in trace['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X']
    assert kernels,'CUPTI returned no GPU kernels; do not infer GPU attribution from CPU-only trace'
    by_name=defaultdict(lambda:dict(count=0,total_us=0.,grids=[]))
    categories=defaultdict(lambda:dict(count=0,total_us=0.))
    for e in kernels:
        name=e['name'];group=kernel_category(name);dur=e['dur']
        by_name[name]['count']+=1;by_name[name]['total_us']+=dur
        grid=e.get('args',{}).get('grid')
        if grid is not None and grid not in by_name[name]['grids']:by_name[name]['grids'].append(grid)
        categories[group]['count']+=1;categories[group]['total_us']+=dur
    for rows in (by_name,categories):
        for row in rows.values():row['ms_per_step']=row['total_us']/1000/steps
    intervals=sorted((e['ts'],e['ts']+e['dur']) for e in kernels)
    first,last=intervals[0];union=0.
    for a,b in intervals[1:]:
        if a>last:union+=last-first;first,last=a,b
        else:last=max(last,b)
    union+=last-first
    return dict(profile_steps=steps,kernel_count=len(kernels),kernel_count_per_step=len(kernels)/steps,
        sum_kernel_ms_per_step=sum(e['dur'] for e in kernels)/1000/steps,
        any_kernel_active_ms_per_step=union/1000/steps,
        profiled_gpu_span_ms_per_step=(max(b for _,b in intervals)-intervals[0][0])/1000/steps,
        categories=dict(categories),
        kernels=sorted([dict(name=k,**v) for k,v in by_name.items()],key=lambda r:r['total_us'],reverse=True),
        note='Kernel durations come from an instrumented trace. Any-kernel-active time is not SM utilization; host gaps may be inflated by the profiler.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',required=True,choices=['baseline','mhc','mhc_no_manifold'])
    p.add_argument('--data-dir',default='../data')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--projection-backend',choices=['official','torch','torch_compile','nvidia','nvidia_precise'],default='official')
    p.add_argument('--connection-backend',choices=['deepseek','nvidia','nvidia_precise'],default='deepseek')
    p.add_argument('--fused-connection-grad',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--wandb-group',default='mhc-profiling-20261030')
    p.add_argument('--timed-steps',type=int,default=20)
    p.add_argument('--collect-after-warmup',action='store_true')
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;seed_all(42)
    config=ConnectionConfig(variant=a.variant,projection_backend=a.projection_backend,
        connection_backend=a.connection_backend,fused_connection_grad=a.fused_connection_grad)
    model=ConnectionModel(config).prepare(compile_blocks=True);opt,wd=optimizer_for(model)
    data=PackedData(a.data_dir);source=json.loads(Path('source.json').read_text())
    props=torch.cuda.get_device_properties(0)
    run_id=uuid.uuid4().hex[:8]
    settings=dict(source=source,model=asdict(config),seed=42,microbatch=16,grad_accum=1,batch_tokens=32768,
        gpu=props.name,gpu_uuid=str(props.uuid),gpu_capability=list(torch.cuda.get_device_capability()),sm_count=props.multi_processor_count,
        nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'),
        parameters=sum(p.numel() for p in model.parameters()),parameter_tensors=len(list(model.parameters())),
        optimizer='native MuonAdamW',warmup_steps=20,normal_timed_steps=a.timed_steps,no_finite_check_timed_steps=20,
        collect_after_warmup=a.collect_after_warmup,
        event_instrumented_steps=4,profiler_steps=2,torch=torch.__version__,cuda=torch.version.cuda,
        note=f'{46+a.timed_steps} temporary diagnostic optimizer steps, no checkpoint/pretraining promotion; finite-check-off is diagnostic only')
    kwargs=dict(project='nanochat',entity='yibozhong657-none',group=a.wandb_group,
                name='profile-'+a.variant+'-'+a.connection_backend+'-'+a.projection_backend,job_type='profiling',id=run_id,config=settings)
    try:run=wandb.init(**kwargs,settings=wandb.Settings(init_timeout=60))
    except wandb.errors.CommError:run=wandb.init(**kwargs,mode='offline')
    json_write(out/'config.json',settings)
    step=0;started=time.perf_counter()
    def perform(fn, *, scheduled=False):
        nonlocal step
        if not scheduled:schedule(opt,step,30518,wd)
        value=fn(model,opt,data,step)
        step+=1
        return value
    try:
        print('Warmup '+a.variant,flush=True)
        for _ in range(20):perform(train_step)
        if a.collect_after_warmup:gc.collect()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(a.timed_steps):
            schedule(opt,step,30518,wd)
            torch.cuda.synchronize();tick=time.perf_counter()
            loss=perform(train_step,scheduled=True);torch.cuda.synchronize()
            times.append((time.perf_counter()-tick)*1000)
        normal=summarize(times)
        memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
        print('Normal step ms '+str(normal['mean']),flush=True)
        times=[]
        for _ in range(20):
            schedule(opt,step,30518,wd)
            torch.cuda.synchronize();tick=time.perf_counter()
            loss=perform(lambda *args:phased_step(*args,finite_check=False),scheduled=True);torch.cuda.synchronize()
            times.append((time.perf_counter()-tick)*1000)
        no_check=summarize(times)
        assert all(torch.isfinite(p).all() and torch.isfinite(p.grad).all() for p in model.parameters())
        events=[]
        for _ in range(4):
            loss,phase_times=perform(lambda *args:phased_step(*args,events=True))
            events.append(phase_times)
        phase_means={axis:{name:statistics.mean(x[axis][name] for x in events)
                          for name in events[0][axis]} for axis in events[0]}
        print('Profiling '+a.variant,flush=True)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False,
                     with_stack=False,profile_memory=False) as prof:
            for _ in range(2):
                loss=perform(lambda *args:phased_step(*args,labels=True))
                torch.cuda.synchronize();prof.step()
        trace=out/'trace.json';prof.export_chrome_trace(str(trace))
        attribution=trace_summary(trace,2)
        averages=[]
        for e in prof.key_averages():
            averages.append(dict(name=e.key,count=e.count,cpu_total_us=e.cpu_time_total,
                self_cpu_us=e.self_cpu_time_total,device_total_us=e.device_time_total,
                self_device_us=e.self_device_time_total))
        json_write(out/'operator-averages.json',averages)
        with trace.open('rb') as src,gzip.open(out/'trace.json.gz','wb') as dst:
            while chunk:=src.read(1024*1024):dst.write(chunk)
        trace.unlink()
        result=dict(complete=True,variant=a.variant,run_id=run_id,source=source,config=settings,
            optimizer_updates=step,training_tokens=step*32768,elapsed_seconds=time.perf_counter()-started,
            normal_ms=normal,no_finite_check_ms=no_check,phase_means=phase_means,phase_samples=events,
            cuda_profile=attribution,last_train_ce=loss,pretraining_launched=False,
            normal_timing_memory=memory,projection_backend=a.projection_backend)
        assert step==46+a.timed_steps
        json_write(out/'summary.json',result)
        run.summary.update(dict(complete=True,optimizer_updates=step,training_tokens=step*32768,
            normal_step_ms=normal['mean'],no_finite_check_step_ms=no_check['mean'],
            cuda_kernel_ms=attribution['sum_kernel_ms_per_step'],kernel_count=attribution['kernel_count_per_step']))
        run.save(str(out/'*.json'),base_path=str(out.parent),policy='now');run.finish()
        print('Completed '+a.variant,flush=True)
    except BaseException:
        run.finish(exit_code=1);raise


if __name__=='__main__':main()
