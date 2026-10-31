"""Audit restart gate, then submit and collect the user's two authorized runs."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[3]
STATE=ROOT/'state.json'


def command(args):
    return subprocess.check_output(args,cwd=ROOT,text=True,timeout=60).strip()


def save(state):
    p=STATE.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2)+'\n');p.replace(STATE)


def job(ident):
    attrs='ClusterId,ProcId,JobStatus,ExitCode,HoldReason,NumJobStarts,RemoteHost'
    rows=json.loads(command(['condor_q',ident,'-json','-attributes',attrs]) or '[]')
    if not rows:
        rows=json.loads(command(['condor_history',ident,'-limit','1','-scanlimit','10000','-json','-attributes',attrs]) or '[]')
    return rows[0] if rows else None


def failure(row):
    return row and (row['JobStatus'] in (3,5) or row['JobStatus']==4 and row.get('ExitCode')!=0)


def read_archive(path,member,*,formal_training=False):
    with tarfile.open(path) as archive:
        value=json.load(archive.extractfile(member))
        offline=[m for m in archive.getmembers() if formal_training and '/offline-run-' in m.name and m.isfile()]
        if offline:
            dest=ROOT/'results'/(path.stem+'-offline')
            for item in offline:
                target=dest/item.name;assert target.resolve().is_relative_to(dest.resolve())
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(item).read())
            for run_dir in sorted((dest/'results/wandb').glob('offline-run-*')):
                command(['wandb','sync',str(run_dir)])
    return value


def verify_gate(state):
    value=read_archive(ROOT/'results'/f'gate-{state["gate_job"]}.tar','results/pretrain-gate.json')
    assert value['complete'] and value['source']==state['gate_source']
    assert value['checks']==dict(tests=3,failures=0,errors=0,skipped=0)
    for variant,trial in value['trials'].items():
        assert trial['complete'] and trial['steps']==50
        assert trial['source']==state['gate_source'] and trial['training_tokens']==50*32768
    value['local_verified']=True
    for path in (ROOT/'gate.json',REPO/'research/connections/pretrain/gate-results.json'):
        path.write_text(json.dumps(value,indent=2)+'\n')
    state.update(gate_verified=True,status='gate_complete_waiting_budget');save(state)


def verify_data(state):
    path=ROOT/'results'/f'data-{state["data_job"]}.tar'
    value=read_archive(path,'results/data.json')
    archive=read_archive(path,'results/data-archive.json')
    assert value['complete'] and value['logging_backend']=='local'
    assert value['requested_tokens']==state['requested_tokens']==10_000_000_000
    assert value['training_steps']==305176 and value['training_tokens']==10_000_007_168
    assert value['train_shards']=='000..012' and value['heldout_shard']=='013'
    assert value['files']['dev']['tokens']==131072 and value['files']['test']['tokens']==2097152
    assert value['first_1b_prefix_sha256']=='614ef0ecce77ca839cb7650bfe7da9a8fbb92bbfc8efe461ded57a1f2af315c7'
    with tarfile.open(path) as report:
        recorded={name:digest for digest,name in (line.split() for line in report.extractfile('results/prepare-inputs.sha256').read().decode().splitlines())}
    assert recorded==state['data_preparation_hashes']
    staging=Path('/staging/y/yzhong239')/state['data_uri'].rsplit('/',1)[-1]
    assert staging.stat().st_size==archive['bytes']
    # Each training worker rehashes every transferred token file before CUDA work.
    state.update(data_ready=True,data_training_tokens=value['training_tokens'],data_archive=archive,
                 status='data_ready_waiting_training_submission')
    (ROOT/'data.json').write_text(json.dumps(value,indent=2)+'\n')
    save(state)


def collect_training(state):
    import wandb
    api=wandb.Api(timeout=30);results={}
    for variant,info in state['training_jobs'].items():
        path=ROOT/'results'/f'{variant}-{info["job"]}.tar'
        value=read_archive(path,'results/train/summary.json',formal_training=True)
        assert value['complete'] and value['source']==state['training_source']
        assert value['steps']==state['training_steps'] and value['test_evaluated']
        assert value['training_tokens']==state['training_steps']*32768
        assert value['run_id']==info['run_id'] and value['test_tokens']==2097152
        run=api.run('yibozhong657-none/nanochat/'+info['run_id'])
        assert run.state=='finished' and run.summary['complete']
        assert math.isclose(run.summary['final_test_ce'],value['final_test_ce'],rel_tol=1e-8)
        checkpoint=ROOT/'results'/f'{variant}-{info["job"]}-checkpoints.tar'
        with tarfile.open(checkpoint) as archive:
            for name in ('checkpoints/latest.pt','checkpoints/best.pt'):
                assert archive.getmember(name).isfile() and archive.getmember(name).size>100_000_000
        value['checkpoint_archive']=str(checkpoint);results[variant]=value
    result=dict(complete=True,wandb_verified=True,trials=results,
        requested_tokens=state['requested_tokens'],source=state['training_source'],
        test_ce_delta_mhc_minus_baseline=results['mhc']['final_test_ce']-results['baseline']['final_test_ce'],
        note='Training may use different GPUs; raw training wall times are not a matched throughput comparison')
    for path in (ROOT/'results.json',REPO/'research/connections/pretrain/results.json'):
        path.write_text(json.dumps(result,indent=2)+'\n')
    state.update(status='complete',push_status='pending_result_review');save(state)


def main():
    with (ROOT/'monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);errors=0
        state_lock=(ROOT/'state.lock').open('w')
        while True:
            # Budget replies must take this same lock before editing state, so
            # a poll cannot overwrite a newly recorded user decision.
            fcntl.flock(state_lock,fcntl.LOCK_EX)
            state=json.loads(STATE.read_text());state['monitor_pid']=os.getpid()
            if state['status'] in ('complete','needs_attention') or state.get('automation_paused_for_wandb_cleanup'):return
            try:
                if state.get('data_job') and not state.get('data_ready'):
                    row=job(state['data_job']);state['last_data_job']=row
                    if failure(row):raise RuntimeError(row)
                    if row and row['JobStatus']==4:verify_data(state)
                if not state.get('gate_verified'):
                    row=job(state['gate_job']);state['last_gate_job']=row
                    if failure(row):raise RuntimeError(row)
                    if row and row['JobStatus']==4:verify_gate(state)
                if state.get('gate_verified') and state.get('budget_confirmed_by_user') and state.get('data_ready') and not state.get('training_jobs'):
                    save(state)
                    print(command([sys.executable,str(REPO/'research/connections/pretrain/submit.py'),str(ROOT)]),flush=True)
                    state=json.loads(STATE.read_text())
                if state.get('training_jobs'):
                    rows={name:job(value['job']) for name,value in state['training_jobs'].items()}
                    state['last_training_jobs']=rows
                    if any(failure(row) for row in rows.values()):raise RuntimeError(rows)
                    if all(row and row['JobStatus']==4 for row in rows.values()):
                        collect_training(state);return
                errors=0;state.update(monitor_errors=0,last_poll=time.time());save(state)
            except Exception as exc:
                errors+=1;state.update(monitor_errors=errors,last_error=repr(exc));print(repr(exc),flush=True)
                if errors>=5 or isinstance(exc,(AssertionError,RuntimeError)):
                    state['status']='needs_attention';save(state);return
                save(state)
            fcntl.flock(state_lock,fcntl.LOCK_UN)
            time.sleep(30)


if __name__=='__main__':main()
