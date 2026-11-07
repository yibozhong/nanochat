"""Production-shape full connector write, including Wg projection and all VJPs."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from torch.profiler import profile,ProfilerActivity
from nanochat.stream_gate_mixing import stream_gate_mix,compiled_stream_gate_reference
from nanochat.nvidia_connections import NvidiaHyperConnection
from nanochat.connection_experiment import configure_attention
from scripts.connection_profile import trace_summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--microbatch',type=int,default=16);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);configure_attention('hardware_auto')
    root=Path('results/stream-gate/micro');root.mkdir(parents=True,exist_ok=True)
    conn=NvidiaHyperConnection(512,4,0,sinkhorn=False).cuda();conn.reset_parameters()
    x=torch.randn(a.microbatch,2048,512,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.randn(a.microbatch,2048,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    post=torch.randn(a.microbatch,2048,4,device='cuda',requires_grad=True)
    h=(torch.randn(a.microbatch,2048,4,4,device='cuda')*.1+torch.eye(4,device='cuda')).requires_grad_()
    w=(torch.randn(4,512,512,device='cuda')*.02).requires_grad_();probe=torch.randn_like(x)
    funcs=dict(hc_write=lambda x,y,p,h:conn.write(y,x,(p,h,None)),
               compiled_torch=compiled_stream_gate_reference,fused=stream_gate_mix)
    results={};traces={}
    for repeat,order in enumerate([list(funcs),list(reversed(funcs))]):
        for name in order:
            fn=funcs[name];args=(x,y,post,h) if name=='hc_write' else (x,y,post,h,w)
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
        args=(x,y,post,h) if name=='hc_write' else (x,y,post,h,w)
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            for _ in range(2):
                grads=torch.autograd.grad(fn(*args),args,probe);del grads
                torch.cuda.synchronize();prof.step()
        path=root/(name+'.trace.json');prof.export_chrome_trace(str(path));traces[name]=trace_summary(path,2);path.unlink()
    result=dict(complete=True,shape=list(x.shape),source=json.loads(Path('source.json').read_text()),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),results=results,traces=traces,optimizer_updates=0,
        includes='Gate RMS, four independent dense projections, sigmoid, HC write and all VJPs; excludes HC route generation, backbone and optimizer')
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
