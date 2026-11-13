"""Fixed pool work distribution sweep, preserving each chunk's equations."""
import json
import os
from pathlib import Path
import torch
import triton as tr
from nanochat import hca_triton as hca

assert os.environ['WANDB_MODE']=='disabled'
torch.manual_seed(741)
xs=[torch.randn(16,2048,4,128,device='cuda',dtype=dtype,requires_grad=True)
    for dtype in (torch.bfloat16,torch.bfloat16,torch.float32)]
upstream=[torch.randn_like(xs[0]) for _ in range(2)]
rows=[];path=Path('results/optimization/pool-sweep.json')
for ratio in (16,4):
    starts=(torch.arange(2048,device='cuda')%ratio==0).float()[None].expand(16,-1)
    hca.FIXED_POOL_CHANNELS,hca.FIXED_POOL_WARPS=32,4
    hca.FIXED_SMALL_POOL_CHANNELS=32
    ref=hca.prefix_pool(*xs,starts,ratio)
    gr=torch.autograd.grad(ref,xs,upstream)
    for channels,warps in [(32,4),(64,4),(128,4),(64,8),(128,8)]:
        hca.FIXED_POOL_CHANNELS,hca.FIXED_POOL_WARPS=channels,warps
        hca.FIXED_SMALL_POOL_CHANNELS=channels
        def run():
            ys=hca.prefix_pool(*xs,starts,ratio)
            return ys,torch.autograd.grad(ys,xs,upstream)
        ys,gs=run()
        for a,e in zip((*ys,*gs),(*ref,*gr)):
            assert torch.isfinite(a).all()
            rel=(a.float()-e.float()).norm()/e.float().norm()
            peak=(a.float()-e.float()).abs().max()/e.float().abs().max()
            assert rel<.02 and peak<.02,(channels,warps,float(rel),float(peak))
        ms=tr.testing.do_bench(run,warmup=100,rep=300)
        row=dict(ratio=ratio,channels=channels,warps=warps,milliseconds=ms,
                 gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
        rows.append(row);print(json.dumps(row),flush=True)
        path.write_text(json.dumps(rows,indent=2)+'\n')
