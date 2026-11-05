"""Collect and publish the implementation gate; never submit training."""
import fcntl
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import time

REPO=Path(__file__).resolve().parents[3]


def cmd(args,cwd=None):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def save(root,state):
    tmp=root/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'state.json')


def collect(root,state,ad):
    archive=root/'results'/('gate-'+state['job']+'.tar')
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            path=Path(member.name)
            assert not path.is_absolute() and '..' not in path.parts
            assert not member.name.endswith(('.pt','.wandb'))
            if member.isfile():
                dest=root/'collected'/path;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(tf.extractfile(member).read())
    assert ad.get('ExitCode')==0,ad
    value=json.loads((root/'collected/results/ffn-width/validation.json').read_text())
    assert value['complete'] and not value['formal_training'] and value['source']==state['source']
    assert value['checks']==dict(tests=28,failures=0,errors=0,skipped=0)
    assert cmd(['git','rev-parse','HEAD'],REPO)==state['source']['commit']
    assert not cmd(['git','status','--porcelain'],REPO)
    report=REPO/'research/connections/ffn-width/validation.json'
    report.write_text(json.dumps(dict(value,job=state['job']),indent=2)+'\n')
    cmd(['git','add',str(report)],REPO)
    cmd(['git','commit','-m','Validate original-MLP FFN-width residual on GPU'],REPO)
    state.update(status='validated_pending_push',reviewed_head=cmd(['git','rev-parse','HEAD'],REPO));save(root,state)


def main(root):
    root=Path(root).resolve()
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            state=json.loads((root/'state.json').read_text())
            if state['status'] in ('complete','needs_attention'):return
            state['monitor_pid']=os.getpid()
            try:
                if state['status']=='validated_pending_push':
                    assert cmd(['git','rev-parse','HEAD'],REPO)==state['reviewed_head']
                    assert not cmd(['git','status','--porcelain'],REPO)
                    # Existing credential-redacting publisher; never copy secrets into jobs.
                    state['pushed_commit']=runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']()
                    state['status']='complete';save(root,state)
                    agents=Path('/home/yzhong239/AGENTS.md')
                    with agents.open('r+') as f:
                        fcntl.flock(f,fcntl.LOCK_EX)
                        text=f.read();heading='## 新实现：FFN-Width Residual（2026-11-04）\n'
                        assert heading in text
                        note=f'\nGPU gate{state["job"]}已exit0：28检查、同卡60步profile及25→50真实恢复通过。main {state["pushed_commit"]}已推送核验；结果research/connections/ffn-width/validation.json。后台收集已完成退出，无正式10B训练。短检查不代表长期效果/稳定性。\n'
                        f.seek(0);f.write(text.replace(heading,heading+note,1));f.truncate()
                    return
                attrs='ClusterId,ProcId,JobStatus,ExitCode,RemoteHost,HoldReason'
                rows=json.loads(cmd(['condor_q',state['job'],'-json','-attributes',attrs]) or '[]')
                if not rows:rows=json.loads(cmd(['condor_history',state['job'],'-limit','1','-json','-attributes',attrs]) or '[]')
                state.update(last_job=rows[0] if rows else None,last_poll=time.time())
                if rows and rows[0]['JobStatus']==4:collect(root,state,rows[0])
                elif rows and rows[0]['JobStatus'] in (3,5):state['status']='needs_attention'
                save(root,state)
            except Exception as exc:
                state.update(status='needs_attention',error=repr(exc));save(root,state);return
            if state['status']=='needs_attention':return
            time.sleep(60)


if __name__=='__main__':main(sys.argv[1])
