"""Single-GPU packed FineWeb-Edu training: baseline or official TileKernels mHC.

Example: python -m scripts.connection_train --variant mhc --steps 50 --data-dir data
Steps are required deliberately; implementation checks do not launch a 1B run.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

import torch
import wandb

from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.packed_training import PackedData, seed_all, evaluate, train_step, snapshot, atomic_save, restore_rng, json_write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=('baseline','mhc','mhc_no_manifold'), required=True)
    p.add_argument('--wandb-group', default='mhc-official-kernels-20261030')
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--schedule-steps', type=int, default=30518)
    p.add_argument('--microbatch', type=int, default=16)
    p.add_argument('--streams', type=int, default=4)
    p.add_argument('--sinkhorn-iterations', type=int, default=20)
    p.add_argument('--activation-checkpoint', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--run-id', default=None)
    p.add_argument('--resume', default=None)
    p.add_argument('--segment-seconds', type=float, default=4*3600)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert a.microbatch>0 and 16%a.microbatch==0
    assert 0<a.steps<=a.schedule_steps
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    ckpt=out/'checkpoints';ckpt.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    seed_all(a.seed)
    c=ConnectionConfig(variant=a.variant,streams=a.streams,sinkhorn_iterations=a.sinkhorn_iterations,
                       activation_checkpoint=a.activation_checkpoint)
    data=PackedData(a.data_dir)
    assert data.manifest['sequence_length']==c.sequence_length
    assert a.steps*16<=len(data.arrays['train'])
    for split, info in data.manifest['files'].items():
        digest=hashlib.sha256()
        with Path(a.data_dir,split+'.bin').open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
        assert digest.hexdigest()==info['sha256'],split
    if Path('source.json').exists():
        source=json.loads(Path('source.json').read_text())
    else:
        source=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    model=ConnectionModel(c).prepare(compile_blocks=True)
    opt,wd=optimizer_for(model)
    protocol=dict(steps=a.steps,schedule_steps=a.schedule_steps,batch_tokens=32768,
                  microbatch=a.microbatch,grad_accum=16//a.microbatch,seed=a.seed)
    step=0;best=float('inf');training_seconds=0.;prior_elapsed=0.
    run_id=a.run_id or uuid.uuid4().hex[:8]
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        assert state['config']==asdict(c) and state['protocol']==protocol
        assert state['data_manifest']==data.manifest and state['source']==source
        if a.run_id:assert state['run_id']==a.run_id
        run_id=state['run_id'];step=state['step'];best=state['best_dev_ce']
        training_seconds=state['training_seconds'];prior_elapsed=state['elapsed_seconds']
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        restore_rng(state['rng']);del state
    config=dict(model=asdict(c),protocol=protocol,data_manifest=data.manifest,source=source,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu_capability=list(torch.cuda.get_device_capability()),
        explicit_portability_gate=os.environ.get('NANOCHAT_MHC_TEST_PORTABILITY')=='1',
        parameters=sum(p.numel() for p in model.parameters()),
        residual_mixing=('sinkhorn' if a.variant=='mhc' else
                         'unconstrained' if a.variant=='mhc_no_manifold' else 'additive'),
        optimizer='native MuonAdamW; connector AdamW at native unembedding LR, no decay',
        optimizer_groups=[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups],
        loss='full-token full-vocabulary CCE; no gradient filtering',
        references='research/connections/references.json')
    kwargs=dict(project='nanochat',entity='yibozhong657-none',group=a.wandb_group,
                job_type='implementation-smoke' if a.steps<=50 else 'pretrain',
                name=f'{a.variant}-{a.steps}steps',id=run_id,resume='allow',config=config)
    try:run=wandb.init(**kwargs,settings=wandb.Settings(init_timeout=60))
    except wandb.errors.CommError:run=wandb.init(**kwargs,mode='offline')
    json_write(out/'config.json',config)
    stop=False;begin=time.perf_counter()
    def interrupted(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGUSR1,interrupted)
    def save():
        atomic_save(snapshot(model,opt,step,config=asdict(c),protocol=protocol,source=source,
            run_id=run_id,data_manifest=data.manifest,best_dev_ce=best,
            training_seconds=training_seconds,elapsed_seconds=prior_elapsed+time.perf_counter()-begin),
            ckpt/'latest.pt')
    try:
        torch.cuda.reset_peak_memory_stats()
        initial_dev,dev_tokens=evaluate(model,data,microbatch=a.microbatch)
        dev=initial_dev
        if step==0:best=initial_dev
        run.log(dict(step=step,tokens=step*32768,**{'dev/ce':initial_dev}))
        durations=[]
        with (out/'steps.jsonl').open('a',buffering=1) as log:
            while step<a.steps:
                lr=schedule(opt,step,a.schedule_steps,wd)
                torch.cuda.synchronize();start=time.perf_counter()
                loss=train_step(model,opt,data,step,microbatch=a.microbatch)
                torch.cuda.synchronize();duration=time.perf_counter()-start
                durations.append(duration);training_seconds+=duration;step+=1
                row=dict(step=step,tokens=step*32768,loss=loss,seconds=duration,lr_multiplier=lr)
                log.write(json.dumps(row)+'\n')
                if step<=5 or step%10==0:
                    print(json.dumps(row),flush=True)
                    run.log(dict(step=step,tokens=row['tokens'],**{'train/ce':loss,'perf/step_seconds':duration}))
                if step==50 or step%250==0 or step==a.steps:
                    dev,dev_tokens=evaluate(model,data,microbatch=a.microbatch)
                    assert math.isfinite(dev);best=min(best,dev)
                    run.log(dict(step=step,tokens=step*32768,**{'dev/ce':dev,'dev/tokens':dev_tokens}))
                if step==50 or step%500==0 or step==a.steps or stop:save()
                if step<a.steps and (stop or time.perf_counter()-begin>=a.segment_seconds):
                    save();run.summary.update(dict(complete=False,checkpoint_step=step))
                    run.finish();return 85
        save()
        result=dict(complete=True,variant=a.variant,run_id=run_id,steps=step,training_tokens=step*32768,
            initial_dev_ce=initial_dev,final_dev_ce=dev,best_dev_ce=best,dev_tokens=dev_tokens,
            training_seconds=training_seconds,elapsed_seconds=prior_elapsed+time.perf_counter()-begin,
            steady_step_seconds=sum(durations[10:])/len(durations[10:]) if len(durations)>10 else None,
            gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),source=source,test_evaluated=False)
        json_write(out/'summary.json',result);run.summary.update(result);run.finish()
        print(json.dumps(result),flush=True);return 0
    except BaseException:
        run.finish(exit_code=1);raise


if __name__=='__main__':
    raise SystemExit(main())
