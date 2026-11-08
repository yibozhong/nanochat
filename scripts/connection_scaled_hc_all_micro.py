"""Production-shape aggregate/write timings with all static scale VJPs."""
import argparse,json,statistics,time
from pathlib import Path
import torch
from torch.profiler import profile,ProfilerActivity
from nanochat.scaled_aggregate import scaled_aggregate
from nanochat.scaled_mixing import scaled_mix
from nanochat.nvidia_ops import nvidia_ops
from scripts.connection_profile import trace_summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--microbatch',type=int,default=16);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    b,t,c=a.microbatch,2048,512
    x=torch.randn(b,t,c,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.randn(b,t,c,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    pre=torch.randn(b,t,4,device='cuda',requires_grad=True);post=torch.randn_like(pre,requires_grad=True)
    h=(torch.randn(b,t,4,4,device='cuda')*.1+torch.eye(4,device='cuda')).requires_grad_()
    scale=(torch.randn(c,4,device='cuda')*.1+1).requires_grad_();ps=scale.detach().clone().requires_grad_()
    def shared(x,p):return nvidia_ops().mhc_fused_aggregate(x,p,4,use_tf32=True,fused_grad_x_acc_buffer=None)
    cases=[('read_shared',shared,[x,pre]),('read_scaled',scaled_aggregate,[x,pre,scale]),
           ('write_mix_only',scaled_mix,[x,y,post,h,scale]),('write_all',scaled_mix,[x,y,post,h,scale,ps])]
    results={};traces={}
    for name,fn,inputs in cases:
        go=torch.randn_like(fn(*inputs))
        for _ in range(5):
            out=fn(*inputs);grad=torch.autograd.grad(out,inputs,go)
        assert all(torch.isfinite(g).all() for g in grad)
        del out,grad
        torch.cuda.synchronize();initial=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(20):
            torch.cuda.synchronize();start=time.perf_counter()
            out=fn(*inputs);grad=torch.autograd.grad(out,inputs,go)
            torch.cuda.synchronize();times.append(time.perf_counter()-start);del out,grad
        results[name]=dict(mean_seconds=statistics.mean(times),step_seconds=times,extra_peak_allocated_bytes=torch.cuda.max_memory_allocated()-initial)
        trace=Path(a.output).with_name(name+'-trace.json')
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            for _ in range(2):
                out=fn(*inputs);grad=torch.autograd.grad(out,inputs,go);torch.cuda.synchronize();del out,grad
        prof.export_chrome_trace(str(trace));traces[name]=trace_summary(trace,2);trace.unlink()
    value=dict(complete=True,shape=[b,t,c,4],warmup=5,timed=20,results=results,traces=traces,
        includes='forward and all VJPs, no optimizer',gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
    Path(a.output).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(complete=True,ms={k:v['mean_seconds']*1000 for k,v in results.items()})),flush=True)


if __name__=='__main__':main()
