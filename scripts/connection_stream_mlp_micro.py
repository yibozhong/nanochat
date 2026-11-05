"""Same-GPU single-connector forward/backward: fused vs compiled PyTorch."""
import gc,json,time
from pathlib import Path
import torch
from nanochat.stream_mlp import stream_mlp


def reference(x,y,p,win,wout):
    h=torch.nn.functional.linear(x.float(),win).relu()
    mixed=torch.nn.functional.linear(h,wout)
    return (mixed+y.float().unsqueeze(-1)*p.unsqueeze(-2)).to(x.dtype)


def main():
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    x=torch.randn(16,2048,512,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.randn(16,2048,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    p=torch.randn(16,2048,4,device='cuda',requires_grad=True)
    win=(torch.randn(16,4,device='cuda')*.5).requires_grad_()
    wout=(torch.randn(4,16,device='cuda')*.25).requires_grad_()
    args=(x,y,p,win,wout);go=torch.randn_like(x)
    results={}
    for name,fn in [('triton_fused',stream_mlp),('torch_compile',torch.compile(reference,fullgraph=True))]:
        def iteration():
            out=fn(*args);return torch.autograd.grad(out,args,go)
        for _ in range(3):iteration()
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(10):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();iteration();end.record();end.synchronize();times.append(start.elapsed_time(end))
        results[name]=dict(mean_ms=sum(times)/len(times),samples_ms=times,
            allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
    root=Path('results/stream-mlp');root.mkdir(parents=True,exist_ok=True)
    (root/'micro.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results),flush=True)
if __name__=='__main__':main()
