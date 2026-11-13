"""Production-sized hard pooling: exact values/VJPs and full forward/back timing."""
import json
import os
from pathlib import Path
import torch
import triton as tr
from nanochat.hca_triton import prefix_pool
from nanochat.hca_parallel_pool import hard_prefix_pool

assert os.environ['WANDB_MODE']=='disabled'
torch.manual_seed(481)
xs=[torch.randn(16,2048,4,128,device='cuda',dtype=dt,requires_grad=True)
    for dt in (torch.bfloat16,torch.bfloat16,torch.float32)]
gs=[torch.randn_like(xs[0]) for _ in range(2)]
rows=[];path=Path('results/optimization/parallel-pool-benchmark.json')
for keep in (0.,1/16,.125):
    starts=(torch.rand(16,2048,device='cuda')<keep).float();starts[:,0]=1
    ref=prefix_pool(*xs,starts);gr=torch.autograd.grad(ref,xs,gs)
    for name,fn in [('serial',prefix_pool),('parallel',hard_prefix_pool)]:
        def run():
            y=fn(*xs,starts)
            return y,torch.autograd.grad(y,xs,gs)
        y,g=run()
        for a,e in zip((*y,*g),(*ref,*gr)):torch.testing.assert_close(a,e,rtol=0,atol=0)
        ms=tr.testing.do_bench(run,warmup=100,rep=300)
        row=dict(keep=keep,implementation=name,milliseconds=ms,
                 exact_forward_vjps=True,gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
        rows.append(row);print(json.dumps(row),flush=True)
        path.write_text(json.dumps(rows,indent=2)+'\n')
