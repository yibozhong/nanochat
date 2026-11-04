"""Frozen packed-data utilities for single-GPU connection experiments."""
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),
        torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(state,path):
    path=Path(path); tmp=path.with_suffix('.tmp')
    torch.save(state,tmp); tmp.replace(path)


def json_write(path,value):
    path=Path(path); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(path)


class PackedData:
    def __init__(self,root='data'):
        self.root=Path(root)
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.arrays={}
        for split,info in self.manifest['files'].items():
            self.arrays[split]=np.memmap(self.root/(split+'.bin'),mode='r',dtype=np.uint16,
                shape=(info['rows'],self.manifest['sequence_length']+1))

    def batch(self,split,start,n,device='cuda'):
        a=self.arrays[split]
        assert start+n <= len(a), (split,start,n,len(a))
        rows=torch.from_numpy(np.array(a[start:start+n],dtype=np.int64,copy=True)).to(device)
        return rows[:,:-1].contiguous(),rows[:,1:].contiguous()


@torch.no_grad()
def evaluate(model,data,split='dev',microbatch=16,max_rows=None,innovation_report=None):
    was_training=model.training
    model.eval()
    total=0.; count=0; sums=None
    limit=len(data.arrays[split]) if max_rows is None else min(max_rows,len(data.arrays[split]))
    for start in range(0,limit,microbatch):
        x,y=data.batch(split,start,min(microbatch,limit-start))
        if innovation_report is not None:
            loss,_,statistics=model(x,y,return_innovation=True,collect_statistics=True)
            sums=statistics if sums is None else sums+statistics
        else:
            loss=model(x,y)
        total += loss.item()*y.numel(); count+=y.numel()
    model.train(was_training)
    if innovation_report is not None:
        from nanochat.innovation_residual import summarize_prediction_sums
        innovation_report.update(beta=model.innovation_beta.item(),
            branches=summarize_prediction_sums(sums,model.config.width))
    return total/count,count


def train_step(model,opt,data,step,microbatch=16,batch_tokens=32768,metrics=None):
    T=model.config.sequence_length
    rows=batch_tokens//T
    assert rows % microbatch == 0
    opt.zero_grad(set_to_none=True)
    total=torch.zeros((),device='cuda')
    innovation=model.config.innovation_residual
    if innovation:
        model.set_innovation_step(step)
        local_total=torch.zeros((),device='cuda')
    for start in range(0,rows,microbatch):
        x,y=data.batch('train',step*rows+start,microbatch)
        if innovation:
            loss,local,_=model(x,y,return_innovation=True)
            # The detach boundaries make these two objectives parameter-disjoint.
            ((loss+local)/(rows//microbatch)).backward()
            local_total += local.detach()/(rows//microbatch)
        else:
            loss=model(x,y)
            (loss/(rows//microbatch)).backward()
        total += loss.detach()/(rows//microbatch)
    # Preserve nanochat's lack of clipping, but stop before a nonfinite update.
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    finite=torch.stack([torch.isfinite(g).all() for g in grads]).all()
    if not bool(finite) or not torch.isfinite(total) or (innovation and not torch.isfinite(local_total)):
        raise FloatingPointError('Nonfinite loss/gradient before optimizer update')
    opt.step()
    if innovation and metrics is not None:
        metrics.update({'innovation/predictor_mse':local_total.item()/(2*model.config.depth),
                        'innovation/beta':model.innovation_beta.item()})
    return total.item()


def snapshot(model,opt,step,**extra):
    return dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,
        rng=rng_state(),**extra)
