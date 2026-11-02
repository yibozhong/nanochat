"""Submit/collect one bounded KV-residual validation; no formal-training mutation."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tarfile
import time

ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[2]
PRIOR=Path('/home/yzhong239/lookback-16k-10b-20261102')
KEY='lookback-kv-residual-20261102'
ATTRS='ClusterId,ProcId,JobStatus,ExitCode,HoldReason,NumJobStarts,RemoteHost,KVLookbackKey,TransferInput'


def command(args,cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def write(path,value):
    p=Path(path);t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def note(message):
    p=Path('/home/yzhong239/AGENTS.md');s=p.read_text()
    heading='## 新请求：Lookback复用当前memory KV更新前一块（2026-11-02）\n'
    if heading in s:p.write_text(s.replace(heading,heading+'\n'+message+'\n',1))


def submit():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'STOPPED.json').exists()
        assert not (ROOT/'state.json').exists() and not (ROOT/'submission-intent.json').exists()
        assert not command(['git','status','--porcelain'],REPO)
        active=json.loads(command(['condor_q','yzhong239','-constraint',f'KVLookbackKey == "{KEY}"','-json','-attributes','ClusterId,ProcId']) or '[]')
        assert not active
        for name in ('bundles','results','logs'):(ROOT/name).mkdir(exist_ok=True)
        bundle=ROOT/'bundles';commit=command(['git','rev-parse','HEAD'],REPO)
        paths=[PRIOR/'train.sh',PRIOR/'train.sub']+[PRIOR/'training-bundles'/n for n in ('source.tar','source.json','inputs.sha256')]
        immutable={str(p):sha(p) for p in paths};write(ROOT/'previous-training-inputs.json',immutable)
        prior=json.loads((PRIOR/'bundles/source.json').read_text())
        wheel='transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
        for name in (wheel,'gate-data.tar'):shutil.copy2(PRIOR/'bundles'/name,bundle/name)
        command(['git','archive','--format=tar','--prefix=nanochat-src/',f'--output={bundle/"source.tar"}',commit],REPO)
        shutil.copy2(REPO/'research/lookback/kv-validate.sh',ROOT/'validate.sh');(ROOT/'validate.sh').chmod(0o755)
        hashes={k:v for k,v in prior['hashes'].items() if k!='source.tar'}
        hashes['source.tar']=sha(bundle/'source.tar')
        for name in (wheel,'gate-data.tar'):assert sha(bundle/name)==hashes[name]
        source=dict(commit=commit,hashes=hashes,purpose='KV-residual Lookback implementation validation',formal_training=False,
            lookback_update='kv_residual',sequence_length=16384,chunk_size=2048,
            launcher_sha256=sha(ROOT/'validate.sh'),previous_training_unchanged=True)
        write(bundle/'source.json',source)
        (bundle/'inputs.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items()))
        # Reuse proven resource policy, replace only the role and input bundle.
        original=(PRIOR/'gate.sub').read_text()
        sub=original.replace('+LookbackKey = "lookback-16k-10b-20261102"',f'+KVLookbackKey = "{KEY}"')
        sub=sub.replace('+JobBatchName = "nanochat-lookback16k-gate"','+JobBatchName = "nanochat-lookback-kv-residual-gate"')
        (ROOT/'gate.sub').write_text(sub)
        command(['condor_submit','-dry-run','gate-dryrun.ad','gate.sub'])
        write(ROOT/'submission-intent.json',dict(source=source,time=time.time()))
        out=command(['condor_submit','-terse','gate.sub']);job=out.split()[0];assert job.endswith('.0'),out
        state=dict(status='gpu_validation_pending',job=job,source=source,submitted_at=time.time(),formal_training=False)
        write(ROOT/'state.json',state)
        rows=json.loads(command(['condor_q',job,'-json','-attributes',ATTRS]));assert len(rows)==1
        assert rows[0]['KVLookbackKey']==KEY and '.netrc' not in rows[0]['TransferInput']
        assert 'file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif' in rows[0]['TransferInput']
        assert all(sha(p)==digest for p,digest in immutable.items())
        write(ROOT/'submission-verified.json',rows)
        note(f'新版KV residual实现已提交本地{commit}，仅GPU验证job{job}已实际提交，25检查/旧新版同卡吞吐/新版25→50真实恢复。工作流{ROOT}；尚未GPU通过、未推送、未启动新版正式训练，旧7669762.0/1继续原快照，inputs哈希已核对不变。')
        print(json.dumps(state,indent=2))


def collect(state):
    with tarfile.open(ROOT/'results'/f'gate-{state["job"]}.tar') as tf:
        assert not any(m.name.endswith(('.pt','.wandb')) for m in tf)
        r=json.load(tf.extractfile('results/kv-lookback-gate.json'))
        evidence={Path(m.name).stem:json.load(tf.extractfile(m)) for m in tf
            if m.isfile() and m.name.startswith('results/kv-lookback-checks/') and m.name.endswith('.json')}
    assert r['complete'] and r['source']==state['source'] and not r['formal_training']
    assert r['checks']==dict(tests=25,failures=0,errors=0,skipped=0)
    assert r['logging_backend']=='local' and len(r['trials'])==4
    for trial in r['trials']:
        assert trial['status']=='fits' and trial['causality_passed'] and len(trial['step_seconds'])==30
        assert trial['peak_reserved_bytes']<int(.9*22000*1024**2)
    resumed=r['resumes']['kv_residual']
    assert resumed['summary']['steps']==50 and resumed['summary']['complete']
    assert resumed['config']['model']['lookback_update']=='kv_residual'
    assert resumed['config']['protocol']['schedule_steps']==305176
    assert resumed['summary']['source']==state['source']
    r.update(job=state['job'],evidence=evidence)
    write(ROOT/'validation.json',r)
    assert command(['git','rev-parse','HEAD'],REPO)==state['source']['commit']
    assert not command(['git','status','--porcelain'],REPO)
    report=REPO/'research/lookback/kv-validation.json';write(report,r)
    command(['git','add',str(report)],REPO)
    command(['git','commit','-m','Validate KV-residual Lookback formulas causality and recovery'],REPO)
    state['report_commit']=command(['git','rev-parse','HEAD'],REPO);write(ROOT/'state.json',state)
    state['published_commit']=runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']()
    state.update(status='complete',summary=r['summary'],completed_at=time.time());write(ROOT/'state.json',state)
    note(f'新版验证job{state["job"]}已exit0，25检查、旧新版同卡profile、新版25→exit85→新进程50恢复通过；实现/报告推送{state["published_commit"]}。新版仅实现验证完成，未新增或切换正式训练。')


def watch():
    with (ROOT/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((ROOT/'state.json').read_text())
        if state['status'] in ('complete','needs_attention','stopped_by_user'):return
        state['monitor_pid']=os.getpid();write(ROOT/'state.json',state)
        errors=0
        while not (ROOT/'STOPPED.json').exists():
            try:
                rows=json.loads(command(['condor_q',state['job'],'-json','-attributes',ATTRS]) or '[]')
                if not rows:rows=json.loads(command(['condor_history',state['job'],'-limit','1','-json','-attributes',ATTRS]) or '[]')
                state.update(last_job=rows[0] if rows else None,last_poll=time.time());write(ROOT/'state.json',state)
                if rows:
                    row=rows[0];assert row['KVLookbackKey']==KEY
                    if row['JobStatus'] in (3,5) or row['JobStatus']==4 and row.get('ExitCode')!=0:raise RuntimeError(row)
                    if row['JobStatus']==4:collect(state);return
                errors=0;state['monitor_errors']=0;write(ROOT/'state.json',state)
            except Exception as exc:
                errors+=1;state.update(last_error=repr(exc),monitor_errors=errors);print(repr(exc),flush=True)
                if errors>=5 or isinstance(exc,(AssertionError,RuntimeError)):
                    state['status']='needs_attention';write(ROOT/'state.json',state);return
                write(ROOT/'state.json',state)
            time.sleep(30)
        state['status']='stopped_by_user';write(ROOT/'state.json',state)


if __name__=='__main__':
    {'submit':submit,'watch':watch}[sys.argv[2]]()
