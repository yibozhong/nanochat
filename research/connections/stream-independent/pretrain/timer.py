"""Five-minute live-job timer; stop after the formal run executes a step."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[4]
ROOT=Path(sys.argv[1]).resolve()
ACTIVE=Path('/home/yzhong239/mhc-stream-independent-20261103/ACTIVE.json')
DATA=Path('/home/yzhong239/nanochat-data-2k-fast-20261102')


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def job_state(job):
    args=['-json','-attributes','ClusterId,ProcId,JobStatus,ExitCode,NumJobStarts,RemoteHost,HoldReason']
    rows=json.loads(subprocess.check_output(['condor_q',job,*args],text=True,timeout=30) or '[]')
    if not rows:
        rows=json.loads(subprocess.check_output(['condor_history',job,'-limit','1',*args],text=True,timeout=30) or '[]')
    return rows[0] if rows else None


def last_progress(path):
    if not path.exists():return None
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-131072));lines=f.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:
            value=json.loads(line)
            if isinstance(value,dict) and ('tokens' in value or 'batches' in value):return value
        except ValueError:pass
    return None


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'timer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (ROOT/'STOPPED.json').exists():
            status=dict(timestamp=time.time(),pid=os.getpid(),interval_seconds=300)
            try:
                GATE=Path(read(ACTIVE)['workflow'])
                cache=read(Path('/home/yzhong239/datasets/fineweb-edu-10bt/current-packed-cache.json'))
                gs=read(GATE/'state.json')
                status.update(data_job=job_state('7685535.0'),gate_job=job_state(gs['job']),
                    cache_ready=bool(cache and cache.get('ready')),gate_status=gs['status'],
                    data_progress=last_progress(DATA/'logs/data.7685535.0.out'))
                if gs['status']=='complete' and not (GATE/'published.json').exists():
                    subprocess.run([sys.executable,str(REPO/'research/connections/stream-independent/pretrain/publish.py'),str(GATE)],cwd=REPO,check=True,timeout=240)
                if status['cache_ready'] and gs['status']=='complete' and (GATE/'published.json').exists() and not (ROOT/'state.json').exists():
                    subprocess.run([sys.executable,str(REPO/'research/connections/stream-independent/pretrain/launch.py'),str(ROOT)],cwd=REPO,check=True,timeout=180)
                train=read(ROOT/'state.json')
                if train:
                    status['training_job']=job_state(train['job'])
                    progress=last_progress(ROOT/f'logs/train.{train["job"]}.out')
                    status['training_progress']=progress
                    row=status['training_job']
                    if row and row['JobStatus']==2 and progress and progress.get('step',0)>=1 and math.isfinite(progress.get('loss',float('nan'))):
                        assert progress['tokens']==progress['step']*32768
                        write(ROOT/'first-step.json',dict(job=train['job'],run_id=train['run_id'],source=train['source'],
                            observed_at=time.time(),live_job=row,progress=progress,formal_training=True))
                        status['status']='training_step_observed_timer_stopped'
                        write(ROOT/'timer-status.json',status);print(json.dumps(status),flush=True);return
                status['status']='monitoring'
            except Exception as exc:
                status.update(status='needs_attention',error=repr(exc))
            status['next_poll_at']=time.time()+300
            write(ROOT/'timer-status.json',status)
            with (ROOT/'timeline.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
            print(json.dumps(status),flush=True)
            time.sleep(300)


if __name__=='__main__':main()
