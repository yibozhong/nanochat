"""Trace the first numerical divergence in a trained full/cached forward."""
import json
import os
from pathlib import Path
import torch
from nanochat import hca, hca_triton
from nanochat.connection_model import ConnectionConfig, build_connection_model
from nanochat.connection_experiment import configure_attention
from nanochat.packed_training import PackedData

assert os.environ['WANDB_MODE'] == 'disabled'
torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
configure_attention('fa2')
state = torch.load('../diagnostic-state/latest.pt',map_location='cpu',weights_only=False)
c=ConnectionConfig(**state['config'])
m=build_connection_model(c).prepare(compile_blocks=False).eval()
m.load_state_dict(state['model']);del state
data=PackedData('../data');ids,_=data.batch('dev',0,1)
full={};cached={i:dict(project=[],pool=[],attention=[]) for i in range(c.depth)}
phase='full';active=0
old_pool=hca_triton.prefix_pool
def record_full_pool(*args,**kwargs):
    result=old_pool(*args,**kwargs)
    full[active]['pool']=tuple(x.clone() for x in result)
    return result
hca_triton.prefix_pool=record_full_pool
old_step=hca.pool_step
def record_step(*args,**kwargs):
    result=old_step(*args,**kwargs)
    cached[active]['pool'].append(tuple(x[:,None].to(torch.bfloat16).clone() for x in result[1:]))
    return result
hca.pool_step=record_step
old_cached_pool=hca_triton.cached_pool
def record_cached_pool(*args,**kwargs):
    result=old_cached_pool(*args,**kwargs)
    cached[active]['pool'].append(tuple(x[:,None].clone() for x in result))
    return result
hca_triton.cached_pool=record_cached_pool
for i,b in enumerate(m.blocks):
    project=b.attn.project;accelerated=b.attn.accelerated;cachefn=b.attn.cached
    def proj(*args,_i=i,_fn=project,**kwargs):
        global active
        active=_i
        result=_fn(*args,**kwargs)
        values=(args[0],*result)
        if phase=='full':full[_i]=dict(project=values)
        else:cached[_i]['project'].append(values)
        return result
    def acc(*args,_i=i,_fn=accelerated,**kwargs):
        result=_fn(*args,**kwargs);full[_i]['attention']=result[0].clone();return result
    def cache(*args,_i=i,_fn=cachefn,**kwargs):
        result=_fn(*args,**kwargs);cached[_i]['attention'].append(result.clone());return result
    b.attn.project=proj;b.attn.accelerated=acc;b.attn.cached=cache
def difference(a,e):
    delta=(a.float()-e.float()); changed=(a!=e).reshape(a.shape[1],-1).any(1).nonzero().flatten()
    return dict(relative_l2=(delta.norm()/e.float().norm().clamp_min(1e-12)).item(),
                normalized_peak=(delta.abs().max()/e.float().abs().max().clamp_min(1.)).item(),
                unequal=int((a!=e).sum()),first_changed_token=None if not len(changed) else int(changed[0]))
with torch.no_grad():
    expected=m.hidden(ids);phase='cache';cache=m.make_kv_cache(1);parts=[];lo=0
    for n in (137,1,511,7,1000,392):
        parts.append(m.hidden(ids[:,lo:lo+n],kv_cache=cache));lo+=n
    rows=[]
    for i in range(c.depth):
        row=dict(layer=i)
        for j,name in enumerate(['input','q','raw_k','local_k','v','score','content']):
            row[name]=difference(torch.cat([x[j] for x in cached[i]['project']],1),full[i]['project'][j])
        for j,name in enumerate(['pool_k','pool_v']):
            row[name]=difference(torch.cat([x[j] for x in cached[i]['pool']],1),full[i]['pool'][j])
        row['attention']=difference(torch.cat(cached[i]['attention'],1),full[i]['attention'])
        rows.append(row)
    report=dict(layers=rows,hidden=difference(torch.cat(parts,1),expected),formal_training=False,
                source=json.loads(Path('source.json').read_text()))
Path('results/optimization/cache-diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
