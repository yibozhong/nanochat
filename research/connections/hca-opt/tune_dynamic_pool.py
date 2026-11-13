"""Pool work-distribution sweep on actual trained sparse/median/dense layers."""
import json
import os
from pathlib import Path
import torch
import triton as tr
from nanochat import hca_triton as hca
from nanochat.connection_model import ConnectionConfig, build_connection_model
from nanochat.connection_experiment import configure_attention
from nanochat.packed_training import PackedData

assert os.environ['WANDB_MODE']=='disabled'
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
torch.manual_seed(418)
configure_attention('fa2')
state=torch.load('../diagnostic-state/latest.pt',map_location='cpu',weights_only=False)
assert state['formal_training'] is False
c=ConnectionConfig(**state['config'])
model=build_connection_model(c).prepare(compile_blocks=False).eval()
model.load_state_dict(state['model']);step=state['step'];del state
data=PackedData('../data');ids,_=data.batch('train',step*16,16)
saved=[];original=hca.prefix_pool
def capture(k,v,score,starts,ratio=0):
    saved.append((k.detach(),v.detach(),score.detach(),starts.detach()))
    return original(k,v,score,starts,ratio)
hca.prefix_pool=capture
with torch.no_grad():model.hidden(ids)
hca.prefix_pool=original
rates=[float(s[-1].mean()) for s in saved]
order=sorted(range(len(rates)),key=lambda i:rates[i])
selected=[order[0],order[len(order)//2],order[-1]]
del model
rows=[];path=Path('results/optimization/dynamic-pool-sweep.json')
for layer in selected:
    *values,starts=saved[layer]
    xs=[x.detach().requires_grad_() for x in values]
    upstream=[torch.randn_like(xs[0]) for _ in range(2)]
    hca.DYNAMIC_POOL_CHANNELS,hca.DYNAMIC_POOL_WARPS=32,4
    ref=hca.prefix_pool(*xs,starts)
    gr=torch.autograd.grad(ref,xs,upstream)
    for channels,warps in [(16,4),(32,4),(64,4),(128,4),(32,8),(64,8),(128,8)]:
        hca.DYNAMIC_POOL_CHANNELS,hca.DYNAMIC_POOL_WARPS=channels,warps
        def run():
            ys=hca.prefix_pool(*xs,starts)
            return ys,torch.autograd.grad(ys,xs,upstream)
        ys,gs=run()
        errors=[]
        for a,e in zip((*ys,*gs),(*ref,*gr)):
            assert torch.isfinite(a).all()
            rel=(a.float()-e.float()).norm()/e.float().norm().clamp_min(1e-7)
            peak=(a.float()-e.float()).abs().max()/e.float().abs().max().clamp_min(1.)
            assert rel <= .02 and peak <= .02,(layer,channels,warps,float(rel),float(peak))
            errors.append(dict(relative=float(rel),peak=float(peak)))
        ms=tr.testing.do_bench(run,warmup=50,rep=200)
        row=dict(layer=layer,hard_keep_rate=rates[layer],channels=channels,warps=warps,
                 milliseconds=ms,errors=errors,gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
        rows.append(row);print(json.dumps(row),flush=True)
        path.write_text(json.dumps(rows,indent=2)+'\n')
