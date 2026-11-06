"""One authorized10B after a fully finite gate; poll300s, then hand off."""
import fcntl
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import time

REPO = Path(__file__).resolve().parents[3]
TRAIN = Path('/home/yzhong239/state-tokens-residual-10b-20261105')


def command(args):
    return subprocess.check_output(args,cwd=REPO,text=True,timeout=60).strip()


def save(root, state):
    temp=root/'state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'state.json')


def collect(root,state):
    validate=runpy.run_path(str(REPO/'research/connections/state-tokens/workflow.py'))['validate_gate']
    with tarfile.open(root/'results'/('gate-'+state['job']+'.tar')) as t:
        report=json.load(t.extractfile('results/state-tokens/validation.json'))
        validate(report,state)
        rows=[json.loads(s) for s in t.extractfile('results/state-tokens/restart/steps.jsonl').read().splitlines()]
        assert [r['step'] for r in rows]==list(range(1,51))
        assert all(math.isfinite(r['loss']) for r in rows)
        config=json.load(t.extractfile('results/state-tokens/restart/config.json'))
        assert config['model']['state_write_residual'] and config['model']['state_attention_backend']=='triton'
        assert config['logging_backend']=='local'
    assert command(['git','rev-parse','HEAD'])==state['automation_head']
    assert not command(['git','status','--porcelain'])
    assert not command(['git','diff','--name-only',state['source']['commit'],'HEAD','--',
                        'nanochat','scripts','research/connections/runtime','research/connections/toolchain'])
    report.update(job=state['job'])
    (root/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    path=REPO/'research/connections/state-tokens/validation.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    command(['git','add',str(path)])
    command(['git','commit','-m','Validate residual state writes and Triton attention on GPU'])
    state.update(status='validated_pending_push',reviewed_head=command(['git','rev-parse','HEAD']))
    save(root,state)


def publish_and_submit(root,state):
    assert command(['git','rev-parse','HEAD'])==state['reviewed_head']
    assert not command(['git','status','--porcelain'])
    if state['status']=='validated_pending_push':
        try:
            runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']()
        except Exception:
            # A PTY timeout may occur after Git successfully pushed.
            assert command(['git','ls-remote','origin','refs/heads/main']).split()[0]==state['reviewed_head']
        assert command(['git','ls-remote','origin','refs/heads/main']).split()[0]==state['reviewed_head']
        state.update(status='published',pushed_commit=state['reviewed_head']);save(root,state)
    subprocess.run([sys.executable,str(REPO/'research/connections/state-tokens/submit_train.py'),str(TRAIN)],
                   cwd=REPO,check=True,timeout=60)
    training=json.loads((TRAIN/'state.json').read_text())
    # This existing watcher is variant-independent and exits at first finite step.
    with (TRAIN/'logs/watch-start.out').open('a') as out,(TRAIN/'logs/watch-start.err').open('a') as err:
        proc=subprocess.Popen([sys.executable,str(REPO/'research/connections/channel-mixing/watch_start.py'),str(TRAIN)],
                              cwd=REPO,stdout=out,stderr=err,start_new_session=True)
    (TRAIN/'monitor.pid').write_text(str(proc.pid)+'\n')
    state.update(status='complete',training_job=training['job'],run_id=training['run_id'],
                 training_monitor_pid=proc.pid,collector_complete=True);save(root,state)
    agents=Path('/home/yzhong239/AGENTS.md')
    with agents.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);text=f.read()
        note=(f'\n## State-token自动10B已提交\n\nGate{state["job"]}全26项、两backend60步无非有限值、25→50恢复通过。'
              f'代码/报告已push {state["pushed_commit"]}；唯一正式{training["job"]}/W&B{training["run_id"]}，'
              f'首步watch PID{proc.pid}/300秒。提交不等于已训练，短测不保证全10B稳定。\n')
        f.seek(0);f.write(text.replace('# Workspace Instructions\n','# Workspace Instructions\n'+note,1));f.truncate()


def main(root):
    root=Path(root).resolve()
    with (root/'collect.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for _ in range(576):
            if (root/'STOPPED.json').exists() or (TRAIN/'STOPPED.json').exists():return
            state=json.loads((root/'state.json').read_text())
            if state['status'] in ('needs_attention','complete'):return
            try:
                assert state['automatic_10b_authorized'] and state['automatic_training_count']==1
                if state['status'] in ('validated_pending_push','published'):
                    publish_and_submit(root,state);return
                attrs=['-json','-attributes','ClusterId,ProcId,JobStatus,ExitCode,HoldReason,RemoteHost']
                rows=json.loads(command(['condor_q',state['job'],*attrs]) or '[]')
                if not rows:rows=json.loads(command(['condor_history',state['job'],'-limit','1',*attrs]) or '[]')
                state.update(last_poll=time.time(),last_job=rows[0] if rows else None)
                if rows and rows[0]['JobStatus']==4:
                    assert rows[0].get('ExitCode')==0, rows[0]
                    collect(root,state);publish_and_submit(root,state);return
                if rows and rows[0]['JobStatus'] in (3,5):raise RuntimeError(rows[0])
                save(root,state)
            except Exception as exc:
                state.update(status='needs_attention',collector_complete=True,error=repr(exc));save(root,state);return
            time.sleep(300)
        state.update(status='needs_attention',collector_complete=True,error='48h collection deadline; no automatic retry')
        save(root,state)


if __name__=='__main__':main(sys.argv[1])
