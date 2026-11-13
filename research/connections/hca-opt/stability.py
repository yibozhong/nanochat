"""Bounded diagnostic training, held-out loss and rate; no formal run/W&B."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import time

p=argparse.ArgumentParser()
p.add_argument('--gradient', choices=['dmc','confidence'],default='confidence')
p.add_argument('--mode',choices=['full','fixed_hca','dynamic_hca'],default='dynamic_hca')
p.add_argument('--block-size',type=int,default=16)
p.add_argument('--ratio-weight',type=float,default=.03)
p.add_argument('--steps',type=int,default=2000)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--label',required=True)
p.add_argument('--save-state',action='store_true')
p.add_argument('--resume')
a=p.parse_args()
assert os.environ['WANDB_MODE']=='disabled'

import torch
from nanochat.connection_model import ConnectionConfig,build_connection_model,optimizer_for,schedule
from nanochat.connection_experiment import configure_attention
from nanochat.packed_training import (PackedData,seed_all,train_step,NonfiniteTrainingError,
                                     snapshot,atomic_save,restore_rng)

torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
configure_attention('fa2');seed_all(a.seed)
c=ConnectionConfig(variant='baseline',streams=1,attention_mode=a.mode,
                   hca_backend='triton',hca_block_size=a.block_size,
                   hca_dynamic_gradient=a.gradient if a.mode=='dynamic_hca' else 'dmc',
                   hca_noise=False,hca_ratio_loss_weight=a.ratio_weight)
model=build_connection_model(c).prepare(compile_blocks=False)
opt,wd=optimizer_for(model)
data=PackedData('../data')
out=Path('results/optimization');out.mkdir(parents=True,exist_ok=True)
report=dict(complete=False,formal_training=False,args=vars(a),config=asdict(c),
            gpu=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
            source=json.loads(Path('source.json').read_text()))
start_step=0
if a.resume:
    state=torch.load(a.resume,map_location='cpu',weights_only=False)
    assert state['config']==asdict(c) and state['seed']==a.seed
    model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
    restore_rng(state['rng']);start_step=state['step']
    report.update(resumed_step=start_step,checkpoint_source=state['source'])
    del state
def write():
    (out/(a.label+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
write();timings=[];last=[]
def validation():
    model.eval()
    rows=[]
    with torch.no_grad():
        limit=min(128,len(data.arrays['dev']))
        for start in range(0,limit,16):
            x,y=data.batch('dev',start,min(16,limit-start))
            ce=model(x,y).item()
            metrics={k:v.item() for k,v in getattr(model,'last_hca_metrics',{}).items()}
            if not metrics:metrics=dict(hard_keep_rate=1.,soft_keep_rate=1.)
            rows.append(dict(ce=ce,**{k:metrics[k] for k in ('hard_keep_rate','soft_keep_rate')}))
    model.train()
    result={k:statistics.mean(row[k] for row in rows) for k in rows[0]}
    result['summary_compression']=1/result['hard_keep_rate']
    result['tokens']=limit*c.sequence_length
    return result
with (out/(a.label+'-steps.jsonl')).open('w',buffering=1) as log:
    for step in range(start_step,a.steps):
        schedule(opt,step,305176,wd)
        torch.cuda.synchronize();start=time.perf_counter()
        try:
            loss=train_step(model,opt,data,step,microbatch=16)
        except NonfiniteTrainingError as e:
            report.update(status='nonfinite',failed_update=step+1,failure=e.report)
            write();raise
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        row=dict(step=step+1,loss=loss,seconds=seconds,**getattr(model,'last_train_metrics',{}))
        log.write(json.dumps(row,allow_nan=False)+'\n')
        if step>=5:timings.append(seconds)
        last.append(row)
        if len(last)>100:last.pop(0)
        if (step+1)%100==0:
            print(json.dumps(row,allow_nan=False),flush=True)
            report.update(status='running',step=step+1,last=row);write()
        if (step+1)%500==0:
            val=validation()
            report.setdefault('validation',[]).append(dict(step=step+1,**val));write()
            print(json.dumps(dict(event='VALIDATION',step=step+1,**val)),flush=True)
        if a.save_state and ((step+1)%100==0 or step+1==a.steps):
            path=Path('../diagnostic-state/latest.pt');path.parent.mkdir(exist_ok=True)
            atomic_save(snapshot(model,opt,step+1,config=asdict(c),seed=a.seed,
                                 source=report['source'],formal_training=False,
                                 next_data_row=(step+1)*16),path)
report.update(complete=True,status='finite_limit',steps=a.steps,last_100=last,
              median_seconds=statistics.median(timings),
              final_train_metrics=getattr(model,'last_train_metrics',{}),
              note='Finite diagnostic trajectory only, not proof of 10B stability or quality.')
write()
