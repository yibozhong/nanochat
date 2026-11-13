"""L2 scheduling ablation with exact equality for unchanged dot products."""
import json
import os
from pathlib import Path
import torch
import triton as tr
from nanochat import hca_projection as projection

assert os.environ['WANDB_MODE']=='disabled'
torch.manual_seed(787)
x=torch.randn(16,2048,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
w=torch.randn(512,512,device='cuda',requires_grad=True)
dy=torch.randn(16,2048,512,device='cuda')
rows=[];path=Path('results/optimization/projection-group-sweep.json')
for group in (1,4,8,16):
    projection.GROUP_M=group
    def run():
        y=projection.score_linear(x,w)
        return y,torch.autograd.grad(y,(x,w),dy)
    y,gs=run()
    if group==1:ref=(y,*gs)
    else:
        for a,e in zip((y,*gs),ref):torch.testing.assert_close(a,e,rtol=0,atol=0)
    ms=tr.testing.do_bench(run,warmup=100,rep=500)
    row=dict(group_m=group,milliseconds=ms,gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
    rows.append(row);print(json.dumps(row),flush=True)
    path.write_text(json.dumps(rows,indent=2)+'\n')
