"""Separate local/completed-summary dKV timing and compiler resource evidence."""
import json
import os
from pathlib import Path
import torch
import triton as tr
from nanochat.hca_triton import _attn_dkv

assert os.environ['WANDB_MODE']=='disabled'
torch.manual_seed(477)
b,t,h,d,w=16,2048,4,128,256
q,k,v,do=[torch.randn(b,t,h,d,device='cuda',dtype=torch.bfloat16) for _ in range(4)]
out=torch.zeros_like(q,dtype=torch.float32)
lse=torch.full((b,h,t),8.,device='cuda')
g=torch.zeros((b,t),device='cuda')
dg=torch.empty((b,t,h),device='cuda')
rows=[]
path=Path('results/optimization/dkv-resources.json')
for ratio in (16,4):
    n=(t-1)//ratio
    for source in (0,2):
        for bm,bn,nw in [(32,32,4),(64,32,4),(32,64,4),(64,64,4),
                         (128,64,4),(64,128,4),(128,64,8),(128,128,8)]:
            for splits in ((1,) if source==0 else (1,2,4,8)):
                if splits==1:
                    dk,dv=[torch.zeros_like(q,dtype=torch.float32) for _ in range(2)]
                else:
                    dk,dv=[torch.zeros((splits,b,n,h,d),device='cuda') for _ in range(2)]
                grid=(tr.cdiv(t if source==0 else n,bn),b*h,splits)
                def run():
                    return _attn_dkv[grid](q,k,v,g,do,out,lse,dk,dv,dg,
                           t,h,d,w,ratio,source,bm,bn,d,n,b,splits,num_warps=nw)
                try:
                    kernel=run()
                    ms=tr.testing.do_bench(run,warmup=50,rep=200)
                except tr.OutOfResources as e:
                    rows.append(dict(ratio=ratio,source=source,tile=[bm,bn,nw],splits=splits,
                                     status='out_of_resources',reason=str(e)))
                    path.write_text(json.dumps(rows,indent=2)+'\n')
                    continue
                row=dict(ratio=ratio,source=source,tile=[bm,bn,nw],splits=splits,
                         milliseconds=ms,registers=kernel.n_regs,spills=kernel.n_spills,
                         shared_bytes=kernel.metadata.shared,
                         gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
                rows.append(row);print(json.dumps(row),flush=True)
                path.write_text(json.dumps(rows,indent=2)+'\n')
