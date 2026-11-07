"""Production-shape fused gate/write versus compiled Torch, forward and all VJPs."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F
from nanochat.gated_mixing import gated_mix, SILU_IDENTITY_BIAS


def reference(x,y,post,h,w,b):
    x=x.float();y=y.float()
    v=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps)
    gate=F.silu(torch.einsum('btck,cik->btci',v,w)+b)
    return (torch.einsum('btci,btij->btcj',x*gate,h)+y.unsqueeze(-1)*post.unsqueeze(-2)).to(torch.bfloat16)


def main():
    p=argparse.ArgumentParser();p.add_argument('--microbatch',type=int,default=16)
    p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    b,t,c=a.microbatch,2048,512
    args=[torch.randn(b,t,c,4,device='cuda',dtype=torch.bfloat16,requires_grad=True),
          torch.randn(b,t,c,device='cuda',dtype=torch.bfloat16,requires_grad=True),
          torch.randn(b,t,4,device='cuda',requires_grad=True),
          (torch.randn(b,t,4,4,device='cuda')*.1+torch.eye(4,device='cuda')).requires_grad_(),
          (torch.randn(c,4,4,device='cuda')*.1).requires_grad_(),
          torch.full((c,4),SILU_IDENTITY_BIAS,device='cuda',requires_grad=True)]
    go=torch.randn_like(args[0]);results={}
    for name,fn in [('fused',gated_mix),('compiled_torch',torch.compile(reference,fullgraph=True))]:
        for _ in range(5):
            out=fn(*args);grads=torch.autograd.grad(out,args,go)
        assert all(torch.isfinite(g).all() for g in grads) and torch.isfinite(out).all()
        del out,grads
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();initial=torch.cuda.memory_allocated()
        elapsed=[]
        for _ in range(20):
            torch.cuda.synchronize();tick=time.perf_counter()
            out=fn(*args);grads=torch.autograd.grad(out,args,go)
            torch.cuda.synchronize();elapsed.append(time.perf_counter()-tick)
            del out,grads
        results[name]=dict(mean_seconds=statistics.mean(elapsed),step_seconds=elapsed,
                           extra_peak_allocated_bytes=torch.cuda.max_memory_allocated()-initial)
    value=dict(complete=True,shape=[b,t,c,4],warmup=5,timed=20,results=results,
               includes='forward and all six input/parameter VJPs; no optimizer',
               gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
    Path(a.output).write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)


if __name__=='__main__':main()
