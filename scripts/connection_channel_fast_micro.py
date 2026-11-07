"""Same-card production-shape timing, all VJPs and real CUDA kernel attribution."""
import json
import os
from pathlib import Path
from functools import partial
import statistics
import time
import torch
from torch.profiler import profile,ProfilerActivity
import triton as tr
from nanochat.channel_mixing import channel_mix
from nanochat.channel_mixing_fast import channel_mix_fast,GEOMETRIES,_channel_fast_forward,_channel_fast_backward
from scripts.connection_profile import trace_summary


def relative(a,b):
    return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)).item()


def main():
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    root=Path('results/channel-fast/micro');root.mkdir(parents=True,exist_ok=True)
    x=torch.randn(16,2048,512,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.randn(16,2048,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    p=torch.randn(16,2048,4,device='cuda',requires_grad=True)
    w=(torch.randn(512,4,4,4,device='cuda')*.5).requires_grad_()
    b=torch.eye(4,device='cuda').expand(512,4,4).contiguous().requires_grad_()
    a=torch.tensor([.17],device='cuda',requires_grad=True)
    args=(x,y,p,w,b,a);go=torch.randn_like(x)
    baseline=channel_mix(*args);bg=torch.autograd.grad(baseline,args,go);baseline=baseline.detach()
    functions={'legacy':channel_mix,**{k:partial(channel_mix_fast,kernel=k) for k in GEOMETRIES}}
    result={}
    for repetition,order in enumerate([list(functions),list(reversed(functions))]):
        for name in order:
            fn=functions[name]
            out=fn(*args);grad=torch.autograd.grad(out,args,go)
            errors=dict(output=relative(out,baseline),gradients=[relative(g,h) for g,h in zip(grad,bg)])
            assert errors['output']<.006 and max(errors['gradients'])<.012,errors
            assert torch.isfinite(out).all() and all(torch.isfinite(g).all() for g in grad)
            del out,grad
            for _ in range(5):torch.autograd.grad(fn(*args),args,go)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();initial=torch.cuda.memory_allocated()
            times=[]
            for _ in range(20):
                torch.cuda.synchronize();tick=time.perf_counter()
                grad=torch.autograd.grad(fn(*args),args,go)
                torch.cuda.synchronize();times.append(time.perf_counter()-tick);del grad
            value=dict(mean_seconds=statistics.mean(times),seconds=times,errors=errors,
                       extra_peak_allocated_bytes=torch.cuda.max_memory_allocated()-initial)
            result.setdefault(name,[]).append(value)
            print(json.dumps(dict(kernel=name,repetition=repetition,**value)),flush=True)
    traces={};resources={}
    for name,fn in functions.items():
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            for _ in range(2):
                grad=torch.autograd.grad(fn(*args),args,go);del grad
                torch.cuda.synchronize();prof.step()
        trace=root/(name+'-trace.json');prof.export_chrome_trace(str(trace))
        traces[name]=trace_summary(trace,2);trace.unlink()
        if name!='legacy':
            ft,fc,fw,bt,bc,bw,fuse=GEOMETRIES[name];n=32768;c=512
            out=torch.empty_like(x);gx=torch.empty_like(x);gy=torch.empty_like(y)
            parts=torch.empty((tr.cdiv(n,bt),c,81),device='cuda')
            pp=torch.empty((n,tr.cdiv(c,bc),4),device='cuda') if fuse else torch.empty_like(p)
            kernels=[_channel_fast_forward[(tr.cdiv(n,ft),tr.cdiv(c,fc))](x,y,p,w,b,a,out,n,c,ft,fc,num_warps=fw),
                     _channel_fast_backward[(tr.cdiv(n,bt),tr.cdiv(c,bc))](x,y,p,w,b,a,go,gx,gy,parts,pp,n,c,bt,bc,fuse,num_warps=bw)]
            resources[name]=[]
            for kind,kernel in zip(['forward','backward'],kernels):
                ptx=kernel.asm['ptx'];(root/(name+'-'+kind+'.ptx')).write_text(ptx)
                resources[name].append(dict(kind=kind,registers=kernel.n_regs,spills=kernel.n_spills,
                    shared_bytes=kernel.metadata.shared,ptx_local_loads=ptx.count('ld.local'),ptx_local_stores=ptx.count('st.local')))
            del out,gx,gy,parts,pp
    summary=dict(complete=True,source=json.loads(Path('source.json').read_text()),shape=list(x.shape),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),results=result,traces=traces,resources=resources,
        optimizer_updates=0,includes='Forward and all six VJPs, no optimizer; instrumented trace separate from ordinary timings')
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
