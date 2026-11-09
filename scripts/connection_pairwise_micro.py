"""Production-shape activation throughput; separate from complete-step timings."""
import json
from pathlib import Path
import torch
from nanochat.pairwise_activation import pairwise_activation,pairwise_torch


def main():
    torch.manual_seed(42)
    x=torch.randn(32768,2048,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    w=(torch.randn(1024,device='cuda')*.1).requires_grad_();go=torch.randn_like(x)
    result={}
    for name,fn,inputs in [('relu2',lambda x,w:x.relu().square(),(x,)),
                           ('torch',pairwise_torch,(x,w)),('triton',pairwise_activation,(x,w))]:
        compiled=torch.compile(fn,fullgraph=True,dynamic=False)
        def execute():torch.autograd.grad(compiled(x,w),inputs,go)
        for _ in range(20):execute()
        torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):execute()
        end.record();end.synchronize();result[name]=start.elapsed_time(end)/100
        torch._dynamo.reset()
    data=dict(gpu=torch.cuda.get_device_name(),shape=list(x.shape),forward_backward_ms=result,
              measurement='CUDA events around100 repeated forward+VJP calls after20warmups; no optimizer; complete-step ABBA reported separately')
    Path('results/pairwise/micro.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data),flush=True)


if __name__=='__main__':main()
