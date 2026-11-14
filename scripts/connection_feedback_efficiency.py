"""Component timings and one real-step CUDA trace, used only by GPU debug."""
from collections import defaultdict
from contextlib import contextmanager,ExitStack
import gzip
import shutil
from unittest.mock import patch

import torch
from torch.nn import functional as F
import triton

from nanochat.layer_feedback import layer_read
from nanochat.projected_feedback import ProjectedLayerReader,normalize_bank
from nanochat.projected_layer_read import projected_read
from nanochat.packed_training import json_write,seed_all


def measure(fn,args,grad_inputs=()):
    compiled=torch.compile(fn,fullgraph=True,dynamic=False)
    for _ in range(3):compiled(*args)
    result=dict(forward_ms=triton.testing.do_bench(lambda:compiled(*args),warmup=50,rep=150))
    if grad_inputs:
        grad=torch.randn_like(compiled(*args))
        def step():return torch.autograd.grad(compiled(*args),grad_inputs,grad)
        for _ in range(3):step()
        result['forward_backward_ms']=triton.testing.do_bench(step,warmup=50,rep=150)
    return result


def component_benchmark():
    seed_all(91);d=512
    x=torch.randn(16,2048,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    bank=torch.randn(16,2048,8,d,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(d,d,device='cuda',requires_grad=True)*.02;k=k.detach().requires_grad_()
    v=torch.randn_like(k,requires_grad=True)
    def linear(a,w):return F.linear(a,w.to(a.dtype))
    def effective_query(a,w):return F.linear(a,w.T.to(a.dtype))
    def materialize(b,kw,vw):return F.linear(b,torch.cat((kw,vw),0).to(b.dtype))
    result=dict(shape=[16,2048,8,d],components={})
    parts=result['components']
    parts['bank_normalization']=measure(normalize_bank,(bank,))
    parts['one_tokenwise_projection']=measure(linear,(x,k),(x,k))
    parts['reassociated_key_projection']=measure(effective_query,(x,k),(x,k))
    parts['materialized_kv_projection']=measure(materialize,(bank,k,v),(k,v))
    kv=materialize(bank,k,v).detach().requires_grad_()
    for backend in ('torch','triton'):
        def direct(q,b):return projected_read(q,b,backend=backend)
        parts[f'materialized_read_{backend}']=measure(direct,(x,kv),(x,kv))
        def compact(q,b):return layer_read(q,b,backend=backend,normalize=False)
        parts[f'reassociated_read_{backend}']=measure(compact,(x,bank),(x,))
    del kv
    for order,backend in (('materialized','triton'),('reassociated','triton'),('reassociated','torch')):
        seed_all(42);reader=ProjectedLayerReader(d,backend,order).cuda()
        parts[f'full_reader_{order}_{backend}']=measure(reader.forward,(x,bank),(x,*reader.parameters()))
    result['reader_speedup']=parts['full_reader_materialized_triton']['forward_backward_ms']/parts['full_reader_reassociated_triton']['forward_backward_ms']
    return result


def annotated(fn,label):
    def call(*args,**kwargs):
        with torch.profiler.record_function(label):return fn(*args,**kwargs)
    return call


@contextmanager
def profile_step(model,opt,path):
    """Wrap a warmed, ordinary training step; instrumentation stays outside graphs."""
    with ExitStack() as stack:
        backbone=model.backbone
        def wrapped_backbone(*args,**kwargs):
            label='stage/backbone_final' if torch.is_grad_enabled() else 'stage/backbone_preliminary'
            with torch.profiler.record_function(label):return backbone(*args,**kwargs)
        stack.enter_context(patch.object(model,'backbone',wrapped_backbone))
        stack.enter_context(patch.object(model,'forward',annotated(model.forward,'stage/forward_loss')))
        stack.enter_context(patch.object(model,'readout',annotated(model.readout,'stage/lm_head')))
        stack.enter_context(patch.object(opt,'step',annotated(opt.step,'stage/optimizer')))
        if getattr(model,'compiled_readers',None) is not None:
            readers=[annotated(fn,f'stage/reader_{i}') for i,fn in enumerate(model.compiled_readers)]
            stack.enter_context(patch.object(model,'compiled_readers',readers))
            stack.enter_context(patch.object(model,'compiled_bank_norm',annotated(model.compiled_bank_norm,'stage/bank_norm')))
        elif getattr(model,'compiled_mix',None) is not None:
            stack.enter_context(patch.object(model,'compiled_mix',annotated(model.compiled_mix,'stage/feedback_mix')))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,profile_memory=True,with_stack=False) as trace:
            yield
            torch.cuda.synchronize()
    events=trace.key_averages(group_by_input_shape=True)
    rows=[dict(name=e.key,count=e.count,cpu_self_us=e.self_cpu_time_total,
        gpu_self_us=e.self_device_time_total,gpu_total_us=e.device_time_total,
        input_shapes=e.input_shapes) for e in events]
    (path.with_suffix('.txt')).write_text(events.table(sort_by='self_device_time_total',row_limit=40))
    kernels=defaultdict(lambda:dict(count=0,gpu_us=0.))
    spans=[]
    for event in trace.events():
        if event.device_type==torch.autograd.DeviceType.CUDA:
            kernels[event.name]['count']+=1;kernels[event.name]['gpu_us']+=event.device_time_total
            spans.append((event.time_range.start,event.time_range.end))
    union=0.;end=None
    for start,stop in sorted(spans):
        union+=max(0.,stop-max(start,end if end is not None else start))
        end=max(stop,end if end is not None else stop)
    elapsed=max(b for a,b in spans)-min(a for a,b in spans) if spans else 0.
    report=dict(stages=[r for r in rows if r['name'].startswith('stage/')],
        top_operators=sorted(rows,key=lambda r:r['gpu_self_us'],reverse=True)[:40],
        copies_and_allocations=[r for r in rows if any(s in r['name'] for s in ('copy','contiguous','clone','cat','empty'))],
        top_cuda_kernels=sorted((dict(name=n,**v) for n,v in kernels.items()),key=lambda r:r['gpu_us'],reverse=True)[:40],
        cuda_kernel_count=sum(k['count'] for k in kernels.values()),device_span_us=elapsed,
        device_active_us=union,device_gap_fraction=(elapsed-union)/elapsed if elapsed else None,
        note='One instrumented step, excluded from throughput averages. Stage GPU times are inclusive and nested; do not sum them. Device gaps include CPU launch and synchronization stalls plus profiling overhead.')
    json_write(path.with_suffix('.json'),report)
    raw=path.with_suffix('.trace.json');trace.export_chrome_trace(str(raw))
    with raw.open('rb') as src,gzip.open(str(raw)+'.gz','wb') as dst:shutil.copyfileobj(src,dst)
    raw.unlink()
