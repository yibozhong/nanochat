"""Single authorized VO16-bucket balanced-hash run with durable resume.

Outer lifecycle follows connection_train; original model/kernel files unchanged.
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

import numpy as np
import torch
from nanochat.experiment_logging import training_run

from nanochat.connection_model import ConnectionConfig, optimizer_for, schedule
from nanochat.hash_vo import HashVOModel
from nanochat.packed_training import PackedData, seed_all, snapshot, atomic_save, restore_rng, json_write
from nanochat.connection_experiment import configure_attention



@torch.no_grad()
def evaluate(model, data, split='dev', microbatch=16):
    was_training = model.training
    model.eval()
    total = 0.; count = 0
    for start in range(0, len(data.arrays[split]), microbatch):
        n = min(microbatch, len(data.arrays[split]) - start)
        x, y = data.batch(split, start, n, device='cpu')
        routing = model.make_routing_plan(x)
        loss = model(x.cuda(), y.cuda(), routing=routing)
        total += loss.item() * y.numel(); count += y.numel()
    model.train(was_training)
    return total / count, count


def train_step(model, opt, data, step, microbatch=16):
    assert microbatch == 16 and model.config.sequence_length == 2048
    opt.zero_grad(set_to_none=True)
    x, y = data.batch('train', step * 16, 16, device='cpu')
    routing = model.make_routing_plan(x)
    loss = model(x.cuda(), y.cuda(), routing=routing)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    finite = torch.stack([torch.isfinite(g).all() for g in gradients]).all()
    if not bool(finite) or not bool(torch.isfinite(loss)):
        raise FloatingPointError(f'Nonfinite hash VO loss/gradient before update {step+1}')
    opt.step()
    return loss.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--formal-training', action='store_true')
    p.add_argument('--wandb-group', default='hash-vo-10b-20261107')
    p.add_argument('--lookup', default='research/connections/hash-qkv/lookup.npy')
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--schedule-steps', type=int, default=30518)
    p.add_argument('--microbatch', type=int, default=16)
    p.add_argument('--activation-checkpoint', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--checkpoint-dir', default=None)
    p.add_argument('--attention-backend', choices=('auto','sdpa_cudnn','fa3','fa2','hardware_auto'), default='auto')
    p.add_argument('--evaluate-test', action='store_true')
    p.add_argument('--run-id', default=None)
    p.add_argument('--resume', default=None)
    p.add_argument('--segment-seconds', type=float, default=1800,
                   help='Exit85 for durable checkpoint upload; worker-local saves are not durable')
    p.add_argument('--segment-steps', type=int, default=0,
                   help='Bounded restart validation; 0 uses only the time/signal limit')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    rows_per_step=16
    assert a.microbatch==16, 'Validated production microbatch is16'
    assert a.microbatch>0 and rows_per_step%a.microbatch==0
    assert 0<a.steps<=a.schedule_steps
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    ckpt=Path(a.checkpoint_dir) if a.checkpoint_dir else out/'checkpoints'
    ckpt.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    actual_attention_backend = configure_attention(a.attention_backend)
    seed_all(a.seed)
    c=ConnectionConfig(activation_checkpoint=a.activation_checkpoint)
    lookup=np.load(a.lookup,allow_pickle=False)
    lookup_sha256=hashlib.sha256(lookup.tobytes()).hexdigest()
    hash_config=dict(kind='vo_hash',buckets=16,top_k=1,lookup_sha256=lookup_sha256,
        routing='frozen_train_frequency_balanced',frequency_training_input_tokens=4194304,
        hash_shared_across_layers=True,weights_shared_across_layers=False,
        qk_fused_execution=True,query_key_projection='shared_dense',value_projection='source_token_hash',
        output_projection='destination_token_hash',initialization='independent_V_zero_O',independent_2d_Muon=True)
    model_config=dict(**asdict(c),hash_vo=hash_config)
    data=PackedData(a.data_dir)
    assert data.manifest['sequence_length']==c.sequence_length
    assert a.steps*rows_per_step<=len(data.arrays['train'])
    for split, info in data.manifest['files'].items():
        digest=hashlib.sha256()
        with Path(a.data_dir,split+'.bin').open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
        assert digest.hexdigest()==info['sha256'],split
    if Path('source.json').exists():
        source=json.loads(Path('source.json').read_text())
    else:
        source=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    assert source['model_variant']=='hash_vo' and source['lookup_sha256']==lookup_sha256
    assert source['initialization']=='seed42_from_zero_no_validation_weights'
    assert a.seed==42
    model=HashVOModel(c,lookup,buckets=16,seed=a.seed).prepare(compile_blocks=True)
    assert sum(p.numel() for p in model.parameters())==121634816
    opt,wd=optimizer_for(model)
    initial_groups=[{k:v for k,v in group.items() if k!='params'} for group in opt.param_groups]
    protocol=dict(steps=a.steps,schedule_steps=a.schedule_steps,batch_tokens=32768,
                  microbatch=a.microbatch,grad_accum=rows_per_step//a.microbatch,seed=a.seed,
                  attention_backend=a.attention_backend,hash_vo=hash_config,
                  evaluate_test=a.evaluate_test)
    step=0;best=float('inf');training_seconds=0.;prior_elapsed=0.;initial_dev=None
    run_id=a.run_id or uuid.uuid4().hex[:8]
    if a.resume=='auto':
        a.resume=str(ckpt/'latest.pt') if (ckpt/'latest.pt').exists() else None
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        assert state['config']==model_config and state['protocol']==protocol
        assert state['data_manifest']==data.manifest and state['source']==source
        if a.run_id:assert state['run_id']==a.run_id
        run_id=state['run_id'];step=state['step'];best=state['best_dev_ce']
        training_seconds=state['training_seconds'];prior_elapsed=state['elapsed_seconds']
        initial_dev=state.get('initial_dev_ce')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        assert np.array_equal(model._lookup_cpu,lookup)
        restore_rng(state['rng']);del state
        assert (ckpt/'best.pt').exists(),'Resume requires the paired dev-best checkpoint'
    start_step=step
    config=dict(logging_backend='wandb' if a.formal_training else 'local',model=model_config,protocol=protocol,data_manifest=data.manifest,source=source,
        checkpoint_transport=dict(upload_interval_seconds=a.segment_seconds,local_save_every_steps=500,
            method='exit85 plus scheduler transfer; local save alone is not durable'),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu_capability=list(torch.cuda.get_device_capability()),
        actual_attention_backend=actual_attention_backend,
        explicit_portability_gate=os.environ.get('NANOCHAT_MHC_TEST_PORTABILITY')=='1',
        parameters=sum(p.numel() for p in model.parameters()),
        residual_mixing='additive',
        attention_projection='balanced_hash_VO_16',
        optimizer='native MuonAdamW; independent2D V/O per expert; shared Q/K; original active-width LR/WD rules',
        optimizer_groups=initial_groups,
        loss='full-token full-vocabulary CCE; no gradient filtering',
        references='research/connections/hash-vo/performance-summary.json')
    from nanochat.connection_cost import model_cost
    cost=model_cost(c,batch=rows_per_step)
    # Same useful active matrix FLOPs; separately account for the validated fixed padding.
    padded_rows=((32768+127)//128+15)*128
    padding_forward=2*(padded_rows-32768)*2*c.width*c.width*c.depth
    cost['vo_padding_forward_flops']=padding_forward
    cost['estimated_forward_flops']+=padding_forward
    cost['estimated_training_flops']+=3*padding_forward
    cost['accounting']+=' HashVO fixed128-row padding included; expanded optimizer cost remains excluded.'
    config['compute_estimate']=cost
    profile_reference=source.get('profile_reference',{}).get('hash_vo')
    if profile_reference:
        config['profile_reference']=profile_reference
        config['profile_matches_current_gpu']=str(torch.cuda.get_device_properties(0).uuid)==profile_reference.get('gpu_uuid')
    kwargs=dict(project='nanochat',entity='yibozhong657-none',group=a.wandb_group,
                job_type='pretrain' if a.formal_training else 'diagnostic',
                name=f'vanilla-vo-hash16-{c.sequence_length}ctx-{a.steps}steps',id=run_id,resume='allow',config=config)
    run=training_run(formal_training=a.formal_training, local_dir=out/"diagnostics", **kwargs)
    run.define_metric('tokens')
    for metric in ('train/*','dev/*','perf/*'):
        run.define_metric(metric,step_metric='tokens')
    json_write(out/'config.json',config)
    stop=False;begin=time.perf_counter()
    def interrupted(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGUSR1,interrupted)
    def save():
        atomic_save(snapshot(model,opt,step,config=model_config,protocol=protocol,source=source,
            run_id=run_id,data_manifest=data.manifest,best_dev_ce=best,
            initial_dev_ce=initial_dev,training_seconds=training_seconds,
            elapsed_seconds=prior_elapsed+time.perf_counter()-begin),
            ckpt/'latest.pt')
        json_write(out/'progress.json',dict(run_id=run_id,step=step,tokens=step*32768,
            best_dev_ce=best,training_seconds=training_seconds,source=source))
    def save_best(dev):
        atomic_save(dict(model=model.state_dict(),config=model_config,protocol=protocol,source=source,
            run_id=run_id,step=step,dev_ce=dev,data_manifest=data.manifest),ckpt/'best.pt')
    try:
        torch.cuda.reset_peak_memory_stats()
        dev,dev_tokens=evaluate(model,data,microbatch=a.microbatch)
        if initial_dev is None:initial_dev=dev
        if step==0:
            best=dev;save_best(dev);save()
        run.log(dict(step=step,tokens=step*32768,**{'dev/ce':dev}))
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
                    perf={'train/ce':loss,'perf/step_seconds':duration,'perf/tokens_per_second':32768/duration,
                          'perf/peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                          'perf/peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                    if cost:
                        perf.update({'perf/estimated_training_flops_per_step':cost['estimated_training_flops'],
                            'perf/estimated_forward_flops_per_step':cost['estimated_forward_flops']})
                    if profile_reference and 'additional_peak_allocated_bytes' in profile_reference:
                        perf['perf/profile_additional_peak_allocated_bytes']=profile_reference['additional_peak_allocated_bytes']
                        perf['perf/profile_additional_peak_reserved_bytes']=profile_reference['additional_peak_reserved_bytes']
                    run.log(dict(step=step,tokens=row['tokens'],**perf))
                if step==50 or step%250==0 or step==a.steps:
                    dev,dev_tokens=evaluate(model,data,microbatch=a.microbatch)
                    assert math.isfinite(dev)
                    if dev<best:
                        best=dev;save_best(dev)
                    run.log(dict(step=step,tokens=step*32768,**{'dev/ce':dev,'dev/tokens':dev_tokens}))
                if step==50 or step%500==0 or step==a.steps or stop:save()
                if step<a.steps and (stop or time.perf_counter()-begin>=a.segment_seconds
                                      or a.segment_steps and step-start_step>=a.segment_steps):
                    save();run.summary.update(dict(complete=False,checkpoint_step=step))
                    run.finish();return 85
        save()
        test_result={}
        if a.evaluate_test:
            final_test,test_tokens=evaluate(model,data,'test',microbatch=a.microbatch)
            selected=torch.load(ckpt/'best.pt',map_location='cpu',weights_only=False)
            assert selected['source']==source and selected['run_id']==run_id
            model.load_state_dict(selected['model'])
            best_test,best_test_tokens=evaluate(model,data,'test',microbatch=a.microbatch)
            assert best_test_tokens==test_tokens
            test_result=dict(final_test_ce=final_test,best_selected_test_ce=best_test,
                best_dev_step=selected['step'],test_tokens=test_tokens)
        checkpoint_files={}
        for name in ('latest.pt','best.pt'):
            path=ckpt/name;digest=hashlib.sha256()
            with path.open('rb') as handle:
                while chunk:=handle.read(8*1024*1024):digest.update(chunk)
            checkpoint_files[name]=dict(bytes=path.stat().st_size,sha256=digest.hexdigest())
        result=dict(complete=True,variant='hash_vo',run_id=run_id,steps=step,training_tokens=step*32768,
            initial_dev_ce=initial_dev,final_dev_ce=dev,best_dev_ce=best,dev_tokens=dev_tokens,
            training_seconds=training_seconds,elapsed_seconds=prior_elapsed+time.perf_counter()-begin,
            steady_step_seconds=sum(durations[10:])/len(durations[10:]) if len(durations)>10 else None,
            gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),source=source,
            test_evaluated=a.evaluate_test,checkpoint_files=checkpoint_files,**test_result)
        json_write(out/'summary.json',result);run.summary.update(result);run.finish()
        print(json.dumps(result),flush=True);return 0
    except BaseException:
        run.finish(exit_code=1);raise


if __name__=='__main__':
    raise SystemExit(main())
