"""Interleave all three models in one process, sharing CCE autotune decisions."""
import importlib
import json
from pathlib import Path
import statistics
import time

import torch
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for,schedule
from nanochat.connection_experiment import configure_attention
from nanochat.ffn_readout import initialize_gradmax
from nanochat.packed_training import seed_all,PackedData,train_step

torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
configure_attention('fa2')
data=PackedData('../data');models={};opts={};wds={}
for name,r,backend in [('baseline',0,'triton'),('reference',64,'reference'),('triton',64,'triton')]:
    seed_all(42)
    c=ConnectionConfig(streams=1,ffn_readout_width=r,ffn_readout_backend=backend)
    models[name]=ConnectionModel(c).prepare(compile_blocks=True)
    if name=='reference':
        x,y=data.batch('train',0,1);initialize_gradmax(models[name],x,y)
    if name=='triton':models[name].load_state_dict(models['reference'].state_dict())
    opts[name],wds[name]=optimizer_for(models[name])
times={name:[] for name in models};losses={name:[] for name in models}
orders=[['baseline','reference','triton'],['triton','reference','baseline'],['reference','baseline','triton']]
for step in range(110):
    for name in orders[step%3]:
        schedule(opts[name],step,305176,wds[name])
        torch.cuda.synchronize();start=time.perf_counter()
        loss=train_step(models[name],opts[name],data,step)
        torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-start)
        if step>=10:times[name].append(elapsed);losses[name].append(loss)
    if step%25==24:print('paired step',step+1,flush=True)
result=dict(formal_training=False,source=json.loads(Path('source.json').read_text()),
    gpu=torch.cuda.get_device_name(),warmup_updates=10,measured_updates=100,
    attention_backend='fa2',shared_autotune=True,alternating_order=orders,models={})
for name in models:
    result['models'][name]=dict(median_ms=statistics.median(times[name]),
        p10_ms=sorted(times[name])[10],p90_ms=sorted(times[name])[90],
        last_loss=losses[name][-1],all_updates_finite=True)
result['paired_triton_minus_reference_ms']=statistics.median([a-b for a,b in zip(times['triton'],times['reference'])])
Path('results/readout/paired.json').write_text(json.dumps(result,indent=2)+'\n')
for mod,attr in [('cce_backward','_cce_backward_kernel'),('cce_lse_forward','_cce_lse_forward_kernel')]:
    kernel=getattr(importlib.import_module('cut_cross_entropy.'+mod),attr)
    while not hasattr(kernel,'cache') and hasattr(kernel,'fn'):
        kernel=kernel.fn
    result[mod+'_configs']={str(k):str(v) for k,v in getattr(kernel,'cache',{}).items()}
Path('results/readout/paired.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
