"""Same-shape existing HC write versus separate eager/compiled self-gating."""
import json
from pathlib import Path
import statistics
import time
import torch
from torch.profiler import profile, ProfilerActivity

from nanochat.self_gated_connections import self_gate, self_gate_reference
from nanochat.nvidia_connections import NvidiaHyperConnection
from nanochat.connection_experiment import configure_attention
from scripts.connection_profile import trace_summary


def main():
    torch.set_num_threads(2);torch.manual_seed(42);configure_attention('hardware_auto')
    root=Path('results/self-gate/micro');root.mkdir(parents=True,exist_ok=True)
    conn=NvidiaHyperConnection(512,4,0,sinkhorn=False).cuda();conn.reset_parameters()
    x=torch.randn(16,2048,512,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.randn(16,2048,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    p=torch.randn(16,2048,4,device='cuda',requires_grad=True)
    h=(torch.randn(16,2048,4,4,device='cuda')*.1+torch.eye(4,device='cuda')).requires_grad_()
    args=(x,y,p,h);probe=torch.randn_like(x)
    funcs=dict(hc_write=lambda x,y,p,h:conn.write(y,x,(p,h,None)),
        eager_self_gate=lambda x,y,p,h:conn.write(y,self_gate_reference(x),(p,h,None)),
        compiled_self_gate=lambda x,y,p,h:conn.write(y,self_gate(x),(p,h,None)))
    results={};traces={}
    for repeat,order in enumerate([list(funcs),list(reversed(funcs))]):
        for name in order:
            fn=funcs[name]
            for _ in range(5):
                out=fn(*args);grads=torch.autograd.grad(out,args,probe)
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
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            for _ in range(2):
                grads=torch.autograd.grad(fn(*args),args,probe);del grads
                torch.cuda.synchronize();prof.step()
        path=root/(name+'.trace.json');prof.export_chrome_trace(str(path));traces[name]=trace_summary(path,2);path.unlink()
    result=dict(complete=True,shape=list(x.shape),formula='silu(x)=x*sigmoid(x)',
        source=json.loads(Path('source.json').read_text()),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        results=results,traces=traces,optimizer_updates=0,includes='Write forward and all four VJPs; no routing generation, branch network or optimizer')
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
