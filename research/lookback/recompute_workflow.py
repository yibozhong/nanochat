"""Validate, publish, then submit exactly one user-authorized 10B rerun model."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tarfile
import time
import uuid

ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[2]
PRIOR=Path('/home/yzhong239/lookback-16k-10b-20261102')
KEY='lookback-shared-kv-10b-20261102'
HEADING='## 新授权：末层共享KV的Lookback，保留self-attention，单组10B（2026-11-02）\n'
ATTRS='ClusterId,ProcId,JobStatus,ExitCode,HoldReason,NumJobStarts,RemoteHost,RecomputeKey,RecomputeRole,RecomputeRunID,TransferInput,SuccessCheckpointExitCode,TransferCheckpoint'
WHEEL='transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
FULL_ARCHIVE='nanochat-16k-10b-data-7667953.tar'
FULL_SHA='d84741d2a06d99d1fd07e33c42ff7e307feba62d4a64166301246ac450eb311c'


def command(args,cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):digest.update(part)
    return digest.hexdigest()


def note(message):
    with Path('/home/yzhong239/AGENTS.md').open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);text=f.read();assert HEADING in text
        f.seek(0);f.write(text.replace(HEADING,HEADING+'\n'+message+'\n',1));f.truncate()


def safe():
    assert not (ROOT/'STOPPED.json').exists(),'Stopped by user'


def query(job):
    rows=json.loads(command(['condor_q',job,'-json','-attributes',ATTRS]) or '[]')
    if not rows:rows=json.loads(command(['condor_history',job,'-limit','1','-json','-attributes',ATTRS]) or '[]')
    if rows:
        assert len(rows)==1 and rows[0]['RecomputeKey']==KEY
        row=rows[0]
        if row['JobStatus'] in (3,5) or row['JobStatus']==4 and row.get('ExitCode')!=0:
            raise RuntimeError(row)
        return row


def baseline_unchanged():
    hashes=json.loads((ROOT/'baseline-inputs.json').read_text())
    assert all(sha(p)==digest for p,digest in hashes.items()),'Baseline inputs changed'


def submit():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);safe()
        assert not (ROOT/'state.json').exists() and not (ROOT/'gate-intent.json').exists()
        assert not command(['git','status','--porcelain'],REPO)
        active=json.loads(command(['condor_q','yzhong239','-constraint',f'RecomputeKey == "{KEY}"','-json','-attributes','ClusterId,ProcId']) or '[]')
        assert not active
        for name in ('bundles','results','logs'):(ROOT/name).mkdir(exist_ok=True)
        paths=[PRIOR/'train.sh',PRIOR/'train.sub']+[PRIOR/'training-bundles'/n for n in ('source.tar','source.json','inputs.sha256')]
        write(ROOT/'baseline-inputs.json',{str(p):sha(p) for p in paths})
        bundle=ROOT/'bundles';commit=command(['git','rev-parse','HEAD'],REPO)
        prior=json.loads((PRIOR/'bundles/source.json').read_text())
        for name in (WHEEL,'gate-data.tar'):shutil.copy2(PRIOR/'bundles'/name,bundle/name)
        command(['git','archive','--format=tar','--prefix=nanochat-src/',f'--output={bundle/"source.tar"}',commit],REPO)
        shutil.copy2(REPO/'research/lookback/recompute-validate.sh',ROOT/'validate.sh');(ROOT/'validate.sh').chmod(0o755)
        hashes={k:v for k,v in prior['hashes'].items() if k!='source.tar'}
        hashes['source.tar']=sha(bundle/'source.tar')
        for name in (WHEEL,'gate-data.tar'):assert sha(bundle/name)==hashes[name]
        source=dict(commit=commit,hashes=hashes,purpose='Shared final-hidden memory with local self-attention and one Transformer refinement',
            formal_training=False,sequence_length=16384,chunk_size=2048,new_learned_parameters=524288,
            variant='lookback_recompute',recompute_input='final_hidden',recompute_memory='shared_final_hidden',
            shared_kv=True,per_layer_kv=False,local_self_attention=True,launcher_sha256=sha(ROOT/'validate.sh'))
        write(bundle/'source.json',source)
        (bundle/'inputs.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items()))
        sub=(PRIOR/'gate.sub').read_text().replace('+LookbackKey = "lookback-16k-10b-20261102"',f'+RecomputeKey = "{KEY}"')
        sub=sub.replace('+LookbackRole','+RecomputeRole').replace('nanochat-lookback16k-gate','nanochat-recompute-gate')
        (ROOT/'gate.sub').write_text(sub);command(['bash','-n','validate.sh'])
        command(['condor_submit','-dry-run','gate-dryrun.ad','gate.sub'])
        write(ROOT/'gate-intent.json',dict(source=source,time=time.time()))
        out=command(['condor_submit','-terse','gate.sub']);job=out.split()[0];assert job.endswith('.0'),out
        state=dict(status='gpu_validation_pending',gate_job=job,source=source,submitted_at=time.time(),
            requested_tokens=10000000000,steps=305176,baseline=dict(job='7669762.0',run_id='0bb03c6e'),
            automatic_continuation=True)
        write(ROOT/'state.json',state)
        rows=json.loads(command(['condor_q',job,'-json','-attributes',ATTRS]));assert len(rows)==1
        assert rows[0]['RecomputeKey']==KEY and '.netrc' not in rows[0]['TransferInput']
        assert 'file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif' in rows[0]['TransferInput']
        baseline_unchanged();write(ROOT/'gate-submission-verified.json',rows)
        note(f'本地源码{commit}已提交GPU gate{job}，18检查、vanilla/共享memory-only/新版lookback三组两轮同卡profile及新版25→50真实恢复；通过后自动审计/push/提交唯一10B正式任务。workflow {ROOT}，尚未GPU通过或正式训练，已完成vanilla7669762.0保持原inputs/结果。')
        print(json.dumps(state,indent=2))


def collect_gate(state):
    archive=ROOT/'results'/f'gate-{state["gate_job"]}.tar'
    with tarfile.open(archive) as tf:
        assert not any(m.name.endswith(('.pt','.wandb')) for m in tf)
        r=json.load(tf.extractfile('results/recompute-gate.json'))
        evidence={Path(m.name).stem:json.load(tf.extractfile(m)) for m in tf
            if m.isfile() and m.name.startswith('results/recompute-checks/') and m.name.endswith('.json')}
    assert r['complete'] and r['source']==state['source'] and not r['formal_training']
    assert r['checks']==dict(tests=18,failures=0,errors=0,skipped=0)
    assert r['logging_backend']=='local' and len(r['trials'])==6
    assert r['new_learned_parameters']==524288 and not r['per_layer_kv'] and r['shared_kv']
    assert r['local_self_attention'] and r['recompute_memory']=='shared_final_hidden'
    assert r['selected_microbatch'] in (1,2)
    for trial in r['trials']:
        assert trial['status']=='fits' and trial['causality_passed'] and len(trial['step_seconds'])==30
        assert trial['peak_reserved_bytes']<int(.9*22000*1024**2)
        assert trial['parameters']==(58720256 if trial['variant']=='baseline' else 59244544)
    resumed=r['resumes']['lookback_recompute']
    assert resumed['summary']['steps']==50 and resumed['summary']['complete']
    assert resumed['config']['model']['recompute_previous']
    assert resumed['config']['model']['recompute_input']=='final_hidden'
    assert resumed['config']['model']['recompute_memory']=='shared_final_hidden'
    assert r['recompute_input']=='final_hidden'
    assert resumed['config']['model']['recompute_chunk_size']==2048
    assert resumed['config']['protocol']['schedule_steps']==305176
    assert resumed['summary']['source']==state['source']
    data=json.loads((PRIOR/'data.json').read_text())
    prefix=json.loads((PRIOR/'gate-data.json').read_text())
    assert r['data_manifest']==prefix and data['sequence_length']==16384
    assert data['files']['train']['rows']>=305176*2
    assert data['first_128_steps_sha256']==prefix['files']['train']['sha256']
    assert data['files']['dev']==prefix['files']['dev'] and data['files']['test']==prefix['files']['test']
    assert sha(PRIOR/'bundles/gate-data.tar')==state['source']['hashes']['gate-data.tar']
    r.update(job=state['gate_job'],evidence=evidence,archive_sha256=sha(archive))
    write(ROOT/'validation.json',r);baseline_unchanged()
    assert command(['git','rev-parse','HEAD'],REPO)==state['source']['commit']
    assert not command(['git','status','--porcelain'],REPO)
    report=REPO/'research/lookback/recompute-validation.json';write(report,r)
    command(['git','add',str(report)],REPO)
    command(['git','commit','-m','Validate chunk rerun causality gradients throughput and recovery'],REPO)
    state.update(status='validated_pending_push',report_commit=command(['git','rev-parse','HEAD'],REPO))
    write(ROOT/'state.json',state)


def publish(state):
    safe();assert command(['git','rev-parse','HEAD'],REPO)==state['report_commit']
    assert not command(['git','status','--porcelain'],REPO)
    state.update(published_commit=runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push'](),status='published_pending_training')
    write(ROOT/'state.json',state)


def submit_training(state):
    safe();baseline_unchanged()
    assert not (ROOT/'training-intent.json').exists(),'Existing submit intent: reconcile before retry'
    assert state['published_commit']==command(['git','rev-parse','HEAD'],REPO)
    report=json.loads((ROOT/'validation.json').read_text());micro=report['selected_microbatch']
    dest=ROOT/'training-bundles';dest.mkdir(exist_ok=True)
    for name in (WHEEL,'source.tar'):shutil.copy2(ROOT/'bundles'/name,dest/name)
    shutil.copy2(REPO/'research/lookback/recompute-train.sh',ROOT/'train.sh');(ROOT/'train.sh').chmod(0o755)
    hashes={k:v for k,v in state['source']['hashes'].items() if k!='gate-data.tar'}
    hashes[FULL_ARCHIVE]=FULL_SHA
    assert sha(dest/'source.tar')==hashes['source.tar']
    old=json.loads((PRIOR/'training-bundles/source.json').read_text())
    assert old['hashes'][FULL_ARCHIVE]==FULL_SHA
    source=dict(commit=state['source']['commit'],hashes=hashes,formal_training=True,
        purpose='User-authorized single10B shared-final-hidden KV and self-attending Transformer refinement',
        sequence_length=16384,chunk_size=2048,variant='lookback_recompute',recompute_input='final_hidden',
        recompute_memory='shared_final_hidden',per_layer_kv=False,shared_kv=True,new_learned_parameters=524288,local_self_attention=True,
        steps=305176,training_tokens=10000007168,gate_job=state['gate_job'],gate_verified=True,
        report_commit=state['published_commit'],profile_reference=report['summary'],launcher_sha256=sha(ROOT/'train.sh'))
    write(dest/'source.json',source);(dest/'inputs.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items()))
    run=uuid.uuid4().hex[:8]
    sub=(PRIOR/'train.sub').read_text().split('queue variant,runid from (')[0]
    sub=sub.replace('+LookbackKey = "lookback-16k-10b-20261102"',f'+RecomputeKey = "{KEY}"')
    sub=sub.replace('+LookbackRole','+RecomputeRole').replace('+LookbackVariant','+RecomputeVariant').replace('+LookbackRunID','+RecomputeRunID')
    sub=sub.replace('nanochat-lookback16k-train','nanochat-recompute10b-train').replace('305176 2',f'305176 {micro}')
    sub+=f'queue variant,runid from (\nlookback_recompute {run}\n)\n'
    (ROOT/'train.sub').write_text(sub)
    command(['bash','-n','train.sh']);command(['condor_submit','-dry-run','train-dryrun.ad','train.sub'])
    write(ROOT/'training-intent.json',dict(source=source,run_id=run,time=time.time()))
    out=command(['condor_submit','-terse','train.sub']);job=out.split()[0];assert job.endswith('.0'),out
    state.update(status='formal_training_submitted',training_job=job,run_id=run,training_source=source,
        selected_microbatch=micro,training_submitted_at=time.time())
    write(ROOT/'state.json',state)
    rows=json.loads(command(['condor_q',job,'-json','-attributes',ATTRS]));assert len(rows)==1
    row=rows[0];assert row['RecomputeRunID']==run and row['RecomputeRole']=='train'
    assert row['SuccessCheckpointExitCode']==85 and row['TransferCheckpoint']=='checkpoints.tar,results.tar'
    assert FULL_ARCHIVE in row['TransferInput'] and '.netrc' in row['TransferInput']
    assert 'file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif' in row['TransferInput']
    write(ROOT/'training-submission-verified.json',rows);baseline_unchanged()
    note(f'GPU gate{state["gate_job"]}已exit0，18检查/三组同卡吞吐/25→50恢复通过；实现报告已推送main {state["published_commit"]}。唯一正式10B新任务{job}/W&B{run}已提交，micro{micro}/accum{2//micro}，不是已开跑。已完成vanilla7669762.0未改，不重复提交。')


def collect_training(state):
    job=state['training_job'];name='lookback_recompute'
    with tarfile.open(ROOT/'results'/f'{name}-{job}.tar') as tf:
        r=json.load(tf.extractfile('results/train/summary.json'))
        c=json.load(tf.extractfile('results/train/config.json'))
    assert r['complete'] and r['variant']==name and r['run_id']==state['run_id']
    assert r['source']==state['training_source'] and c['source']==state['training_source']
    assert r['steps']==305176 and r['training_tokens']==10000007168
    assert r['test_evaluated'] and r['dev_tokens']==131072 and r['test_tokens']==2097152
    gate=json.loads((ROOT/'validation.json').read_text())
    assert c['model']==gate['resumes'][name]['config']['model']
    assert c['data_manifest']==json.loads((PRIOR/'data.json').read_text()) and c['logging_backend']=='wandb'
    micro=state['selected_microbatch']
    assert c['protocol']==dict(steps=305176,schedule_steps=305176,batch_tokens=32768,microbatch=micro,
        grad_accum=2//micro,seed=42,attention_backend='sdpa_cudnn',matched_backbone_init=True,evaluate_test=True)
    with tarfile.open(ROOT/'results'/f'{name}-{job}-checkpoints.tar') as tf:
        assert {m.name for m in tf if m.isfile()}=={'checkpoints/latest.pt','checkpoints/best.pt'}
        for name,meta in r['checkpoint_files'].items():assert tf.getmember('checkpoints/'+name).size==meta['bytes']
    import wandb
    run=wandb.Api(timeout=30).run('yibozhong657-none/nanochat/'+state['run_id'])
    assert run.config['source']==state['training_source']
    if run.state!='finished':raise ValueError('Waiting for W&B finished state')
    for key in ('steps','training_tokens','final_dev_ce','final_test_ce','best_selected_test_ce'):
        if not math.isclose(run.summary.get(key,float('nan')),r[key],rel_tol=1e-8,abs_tol=1e-10):
            raise ValueError('Waiting for W&B summary '+key)
    r.update(config=c,wandb_verified=True,checkpoint_sizes_verified=True)
    write(ROOT/'results/training-verified.json',r)
    state.update(status='complete',training_result=r,completed_at=time.time());write(ROOT/'state.json',state)
    note(f'重算Lookback正式{job}/W&B{state["run_id"]}已正常exit0，10B完成；test CE{r["final_test_ce"]}，W&B与checkpoint审计通过，结果{ROOT}/results/training-verified.json。')


def watch():
    with (ROOT/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((ROOT/'state.json').read_text())
        if state['status'] in ('complete','needs_attention','stopped_by_user'):return
        state['monitor_pid']=os.getpid();write(ROOT/'state.json',state);errors=0
        while not (ROOT/'STOPPED.json').exists():
            try:
                if state['status']=='gpu_validation_pending':
                    row=query(state['gate_job']);state['last_gate']=row
                    if row and row['JobStatus']==4:collect_gate(state)
                if state['status']=='validated_pending_push':publish(state)
                if state['status']=='published_pending_training':submit_training(state)
                if state['status']=='formal_training_submitted':
                    row=query(state['training_job']);state['last_training']=row
                    if row and row['JobStatus']==4:collect_training(state);return
                errors=0;state.update(monitor_errors=0,last_poll=time.time());write(ROOT/'state.json',state)
            except Exception as exc:
                errors+=1;state.update(last_error=repr(exc),monitor_errors=errors);print(repr(exc),flush=True)
                if errors>=5 or isinstance(exc,(AssertionError,RuntimeError)):
                    state['status']='needs_attention';write(ROOT/'state.json',state);return
                write(ROOT/'state.json',state)
            time.sleep(30)
        state['status']='stopped_by_user';write(ROOT/'state.json',state)


if __name__=='__main__':{'submit':submit,'watch':watch}[sys.argv[2]]()
