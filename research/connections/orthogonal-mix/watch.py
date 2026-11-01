"""Audit one gate, publish, release the orthogonal10B run, and collect results."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import time

ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[3]
MODES=('orthogonal',)


def command(args,cwd=ROOT):return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def save(s):
    p=ROOT/'state.tmp';p.write_text(json.dumps(s,indent=2)+'\n');p.replace(ROOT/'state.json')


def job(ident):
    attrs='ClusterId,ProcId,JobStatus,ExitCode,HoldReason,HoldReasonCode,RemoteHost,OrthogonalMixKey,OrthogonalMixRole,ReadoutMixing'
    rows=json.loads(command(['condor_q',ident,'-json','-attributes',attrs]) or '[]')
    if not rows:rows=json.loads(command(['condor_history',ident,'-limit','1','-scanlimit','10000','-json','-attributes',attrs]) or '[]')
    if rows:assert rows[0]['OrthogonalMixKey']=='mhc-orthogonal-mix-10b-20261031'
    return rows[0] if rows else None


def immutable_inputs():
    r=json.loads((ROOT/'immutable-reference-inputs.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in r.items())


def gate(s):
    path=ROOT/'results'/f'gate-{s["gate_job"]}.tar'
    with tarfile.open(path) as tar:
        r=json.load(tar.extractfile('results/orthogonal-mix-gate.json'))
        assert r['complete'] and r['source']==s['source'] and r['logging_backend']=='local'
        assert r['checks']==dict(tests=26,failures=0,errors=0,skipped=0)
        assert set(r['profiles'])==set(r['trials'])==set(r['configs'])==set(MODES)
        assert not any(m.name.endswith(('.pt','.wandb')) or 'wandb' in Path(m.name).parts for m in tar)
        r['unit_results']={Path(m.name).name:json.load(tar.extractfile(m)) for m in tar
            if m.isfile() and m.name.startswith('results/orthogonal-mix-checks/') and m.name.endswith('.json')}
        assert r['unit_results']['initialization.json']['shared_parameters_bitwise_equal']
        assert r['unit_results']['initialization.json']['backbone_sinkhorn_unchanged']
    assert len({x['gpu'] for x in r['profiles'].values()})==1
    for mode in MODES:
        p=r['profiles'][mode];t=r['trials'][mode];c=r['configs'][mode]
        assert p['timed_steps']==30 and math.isfinite(p['mean_seconds']) and p['peak_reserved_bytes']<.9*22000*1024**2
        assert t['complete'] and t['steps']==50 and t['training_tokens']==50*32768 and t['final_dev_ce']<t['initial_dev_ce']
        assert c['model']['partition_mixing']==mode and c['protocol']['schedule_steps']==305176
        assert c['applied_sinkhorn_iterations']==20 and c['logging_backend']=='local'
        if mode=='orthogonal':
            assert t['final_stream_mixing']['orthogonality_max_error']<3e-6
            assert t['final_stream_mixing']['row_sum_max_error']<3e-6
    immutable_inputs();r['gate_job']=s['gate_job'];r['archive_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/'gate.json').write_text(json.dumps(r,indent=2)+'\n')
    s.update(gate_verified=True,status='gate_passed_training_held');save(s)


def publish_release(s):
    if not s.get('report_committed'):
        assert command(['git','rev-parse','HEAD'],REPO)==s['reviewed_head']
        assert not command(['git','status','--porcelain'],REPO)
        report=REPO/'research/connections/orthogonal-mix/gate-results.json';report.write_bytes((ROOT/'gate.json').read_bytes())
        command(['git','add',str(report)],REPO)
        command(['git','commit','-m','Validate orthogonal readout and10B-schedule restart'],REPO)
        s.update(report_committed=True,reviewed_head=command(['git','rev-parse','HEAD'],REPO));save(s)
    if not s.get('pushed_commit'):
        assert command(['git','rev-parse','HEAD'],REPO)==s['reviewed_head']
        assert not command(['git','status','--porcelain'],REPO)
        helper=runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))
        s['pushed_commit']=helper['push']();save(s)
    for mode in MODES:
        t=s['training_jobs'][mode]
        if t['released']:continue
        immutable_inputs();row=job(t['job'])
        assert row and row['OrthogonalMixRole']=='formal' and row['ReadoutMixing']==mode
        if row['JobStatus']==5:
            assert row['HoldReasonCode']==15,row
            command(['condor_release',t['job']])
        else:assert row['JobStatus'] in (1,2,4),row
        t.update(released=True,released_at=time.time());save(s)
    if s['status']!='training_released':
        s['status']='training_released';save(s)
        p=Path('/home/yzhong239/AGENTS.md')
        with p.open('r+') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            text=handle.read();heading='## 最新纠正：只跑正交mixing，直接10B（2026-10-31）\n'
            note=f'\n最新：gate{s["gate_job"]}已exit0，26检查、正交版吞吐与25→50真实恢复通过；实现/报告已推送main {s["pushed_commit"]}。唯一正式10B作业已自动release：'+', '.join(f'{m}={t["job"]}/W&B{t["run_id"]}' for m,t in s['training_jobs'].items())+'。release不等于已开跑，监控继续跟踪，勿重复提交。\n'
            assert heading in text
            handle.seek(0);handle.write(text.replace(heading,heading+note,1));handle.truncate()


def collect(s,mode):
    t=s['training_jobs'][mode];path=ROOT/'results'/f'{mode}-{t["job"]}.tar'
    with tarfile.open(path) as tar:
        r=json.load(tar.extractfile('results/train/summary.json'));c=json.load(tar.extractfile('results/train/config.json'))
        offline=[m for m in tar if m.isfile() and '/offline-run-' in m.name]
        if offline:
            dest=ROOT/'results'/('offline-'+mode)
            for member in offline:
                target=dest/member.name;assert target.resolve().is_relative_to(dest.resolve())
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(member).read())
            for run in sorted((dest/'results/wandb').glob('offline-run-*')):command(['wandb','sync',str(run)])
    assert r['complete'] and r['source']==s['source'] and r['run_id']==t['run_id']
    assert r['steps']==305176 and r['training_tokens']==10000007168
    assert r['test_evaluated'] and r['test_tokens']==2097152 and r['final_stream_mixing']['tokens']==4096
    assert c['model']['partition_mixing']==mode and c['protocol']['schedule_steps']==305176 and c['logging_backend']=='wandb'
    assert {k:v['sha256'] for k,v in c['data_manifest']['files'].items()}==s['source']['data_split_sha256']
    archive=ROOT/'results'/f'{mode}-{t["job"]}-checkpoints.tar'
    with tarfile.open(archive) as tar:
        for name in ['latest.pt','best.pt']:assert tar.getmember('checkpoints/'+name).size==r['checkpoint_files'][name]['bytes']
    import wandb
    run=wandb.Api(timeout=30).run('yibozhong657-none/nanochat/'+t['run_id'])
    if run.state!='finished' or not math.isclose(run.summary.get('final_test_ce',float('nan')),r['final_test_ce'],rel_tol=1e-8):
        raise ValueError('Waiting for formal W&B summary synchronization')
    r.update(wandb_verified=True,checkpoint_archive=str(archive),partition_mixing=mode)
    t.update(collected=True,result=r);save(s)


def main():
    if (ROOT/'STOPPED.json').exists():return
    with (ROOT/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);errors=0
        while True:
            s=json.loads((ROOT/'state.json').read_text());s['monitor_pid']=os.getpid()
            if s['status'] in ('complete','needs_attention','stopped_by_user') or (ROOT/'STOPPED.json').exists():return
            try:
                if not s['gate_verified']:
                    row=job(s['gate_job']);s['last_gate_job']=row
                    if row and (row['JobStatus'] in (3,5) or row['JobStatus']==4 and row.get('ExitCode')!=0):raise RuntimeError(row)
                    if row and row['JobStatus']==4:gate(s)
                if s['gate_verified']:publish_release(s)
                for mode,t in s['training_jobs'].items():
                    if t['collected']:continue
                    row=job(t['job']);t['last_job']=row
                    if row and (row['JobStatus']==3 or row['JobStatus']==4 and row.get('ExitCode')!=0):raise RuntimeError(row)
                    if row and row['JobStatus']==5 and (t['released'] or row.get('HoldReasonCode')!=15):raise RuntimeError(row)
                    if t['released'] and row and row['JobStatus']==4:collect(s,mode)
                if all(t['collected'] for t in s['training_jobs'].values()):
                    result=dict(complete=True,steps=305176,training_tokens_per_run=10000007168,
                        trials={m:t['result'] for m,t in s['training_jobs'].items()})
                    (ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
                    s['status']='complete'
                s.update(last_poll=time.time(),monitor_errors=0);save(s);errors=0
                if s['status']=='complete':return
            except Exception as exc:
                errors+=1;s.update(last_error=repr(exc),monitor_errors=errors)
                print(repr(exc),flush=True)
                if errors>=5 or isinstance(exc,(AssertionError,RuntimeError)):s['status']='needs_attention'
                save(s)
                if s['status']=='needs_attention':return
            time.sleep(30)


if __name__=='__main__':main()
