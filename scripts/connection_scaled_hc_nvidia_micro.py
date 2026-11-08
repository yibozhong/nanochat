"""Production-shape scale/write correctness, timing and kernel attribution."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from torch.profiler import profile, ProfilerActivity
from nanochat.scaled_mixing import scaled_mix
from scripts.connection_scaled_hc_nvidia_profile import legacy_function
from nanochat.scaled_nvidia_kernels import _nvidia_scaled_fwd,_nvidia_scaled_bwd
from nanochat.nvidia_ops import nvidia_ops
from scripts.connection_profile import trace_summary


def reference(x,y,post,h,a):
    return (torch.einsum('btci,btij->btcj',x.float()*a,h)+y.float().unsqueeze(-1)*post.unsqueeze(-2)).to(x.dtype)


def relative(a,b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()


def main():
    p=argparse.ArgumentParser();p.add_argument('--microbatch',type=int,default=16)
    p.add_argument('--output',required=True);options=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    b,t,c=options.microbatch,2048,512
    args=[torch.randn(b,t,c,4,device='cuda',dtype=torch.bfloat16,requires_grad=True),
          torch.randn(b,t,c,device='cuda',dtype=torch.bfloat16,requires_grad=True),
          torch.randn(b,t,4,device='cuda',requires_grad=True),
          (torch.randn(b,t,4,4,device='cuda')*.1+torch.eye(4,device='cuda')).requires_grad_(),
          (torch.randn(c,4,device='cuda')*.1+1).requires_grad_()]
    go=torch.randn_like(args[0]);results={};traces={}
    compiled=torch.compile(reference,fullgraph=True)
    expected=compiled(*args);eg=torch.autograd.grad(expected,args,go)
    out=scaled_mix(*args);grads=torch.autograd.grad(out,args,go)
    errors=dict(output=relative(out,expected),gradients=[relative(x,y) for x,y in zip(grads,eg)])
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    del out,grads,expected,eg
    def shared(x,y,post,h):
        return nvidia_ops().mhc_fused_expand_combine(y,None,post,x,h,4,
            use_tf32=True,fused_grad_x_acc_buffer=None)
    for name,fn,inputs in [('shared_hc',shared,args[:4]),('legacy',legacy_function(),args),('nvidia',scaled_mix,args),('compiled_torch',compiled,args)]:
        for _ in range(5):
            out=fn(*inputs);grads=torch.autograd.grad(out,inputs,go)
        assert all(torch.isfinite(g).all() for g in grads) and torch.isfinite(out).all()
        del out,grads
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();initial=torch.cuda.memory_allocated()
        elapsed=[]
        for _ in range(20):
            torch.cuda.synchronize();tick=time.perf_counter()
            out=fn(*inputs);grads=torch.autograd.grad(out,inputs,go)
            torch.cuda.synchronize();elapsed.append(time.perf_counter()-tick)
            del out,grads
        results[name]=dict(mean_seconds=statistics.mean(elapsed),step_seconds=elapsed,
            extra_peak_allocated_bytes=torch.cuda.max_memory_allocated()-initial)
        trace=Path(options.output).with_name(name+'-trace.json')
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            for _ in range(2):
                out=fn(*inputs);grads=torch.autograd.grad(out,inputs,go)
                torch.cuda.synchronize()
                del out,grads
        prof.export_chrome_trace(str(trace));traces[name]=trace_summary(trace,2);trace.unlink()
    value=dict(complete=True,shape=[b,t,c,4],warmup=5,timed=20,results=results,
        errors=errors,traces=traces,nvidia_configs=dict(forward=str(_nvidia_scaled_fwd.best_config),backward=str(_nvidia_scaled_bwd.best_config)),parameter_workspace_bytes=dict(legacy=4*1024**2,nvidia=64*1024**2),includes='forward and all input/parameter VJPs; no optimizer',
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
    Path(options.output).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(complete=True,ms={k:v['mean_seconds']*1000 for k,v in results.items()},errors=errors)),flush=True)


if __name__=='__main__':main()
