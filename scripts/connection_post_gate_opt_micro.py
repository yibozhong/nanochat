"""Production-shape full connector write, including Wg projection and all VJPs."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from torch.profiler import profile,ProfilerActivity
from nanochat.stream_gate_mixing import stream_gate_mix
from nanochat.post_stream_gate_mixing import post_stream_gate_mix,compiled_post_stream_gate_reference,_post_backward_rowwise
from nanochat.stream_gate_mixing import project_gates
from scripts.connection_post_gate_opt_profile import legacy_module
import triton as tr
from nanochat.connection_experiment import configure_attention
from scripts.connection_profile import trace_summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--microbatch',type=int,default=16);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);configure_attention('hardware_auto')
    root=Path('results/post-gate-opt/micro');root.mkdir(parents=True,exist_ok=True)
    x=torch.randn(a.microbatch,2048,512,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.randn(a.microbatch,2048,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    post=torch.randn(a.microbatch,2048,4,device='cuda',requires_grad=True)
    h=(torch.randn(a.microbatch,2048,4,4,device='cuda')*.1+torch.eye(4,device='cuda')).requires_grad_()
    w=(torch.randn(4,512,512,device='cuda')*.02).requires_grad_();probe=torch.randn_like(x)
    funcs=dict(pre=stream_gate_mix,legacy=legacy_module().post_stream_gate_mix,optimized=post_stream_gate_mix,torch=compiled_post_stream_gate_reference)
    # Full production-shape old/new comparison includes all five VJPs.
    args=(x,y,post,h,w)
    original=funcs['legacy'](*args);og=torch.autograd.grad(original,args,probe)
    candidate=funcs['optimized'](*args);cg=torch.autograd.grad(candidate,args,probe)
    def relative(a,b):return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-14)).item()
    errors=dict(output=relative(candidate,original),gradients=[relative(a,b) for a,b in zip(cg,og)])
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    del original,candidate,og,cg
    results={};traces={}
    for repeat,order in enumerate([list(funcs),list(reversed(funcs))]):
        for name in order:
            fn=funcs[name];args=(x,y,post,h,w)
            for _ in range(5):out=fn(*args);grads=torch.autograd.grad(out,args,probe)
            assert torch.isfinite(out).all() and all(torch.isfinite(g).all() for g in grads)
            del out,grads
            torch.cuda.synchronize();initial=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(20):
                torch.cuda.synchronize();start=time.perf_counter();out=fn(*args)
                grads=torch.autograd.grad(out,args,probe);torch.cuda.synchronize();times.append(time.perf_counter()-start)
                del out,grads
            item=dict(mean_seconds=statistics.mean(times),seconds=times,extra_peak_allocated_bytes=torch.cuda.max_memory_allocated()-initial)
            results.setdefault(name,[]).append(item);print(json.dumps(dict(name=name,repeat=repeat,**item)),flush=True)
    for name,fn in funcs.items():
        args=(x,y,post,h,w)
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            for _ in range(2):
                grads=torch.autograd.grad(fn(*args),args,probe);del grads
                torch.cuda.synchronize();prof.step()
        path=root/(name+'.trace.json');prof.export_chrome_trace(str(path));traces[name]=trace_summary(path,2);path.unlink()
    # Inspect the exact launched kernels, not a separately guessed specialization.
    c=512;n=x.numel()//(4*c);z,r=project_gates(x,w)
    direct=torch.empty_like(x,dtype=torch.float32);gz=torch.empty_like(z)
    gy=torch.empty_like(y);gp=torch.empty_like(post);gh=torch.empty_like(h)
    resource={}
    for name,fn,warps in [('legacy',legacy_module()._post_backward,8),('optimized',_post_backward_rowwise,4)]:
        kernel=fn[(n,)](x,y,post,h,z,probe,direct,gz,gy,gp,gh,n,c,tr.next_power_of_2(c),num_warps=warps)
        torch.cuda.synchronize()
        resource[name]=dict(registers=kernel.n_regs,spills=kernel.n_spills,shared_bytes=kernel.metadata.shared,num_warps=warps)
    result=dict(complete=True,shape=list(x.shape),production_shape_errors=errors,kernel_resources=resource,source=json.loads(Path('source.json').read_text()),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),results=results,traces=traces,optimizer_updates=0,
        includes='Gate RMS, four independent dense projections, sigmoid, HC write and all VJPs; excludes HC route generation, backbone and optimizer')
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
