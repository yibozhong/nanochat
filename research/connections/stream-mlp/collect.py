"""Audit a passed GPU gate, publish and submit the authorized10B exactly once."""
import fcntl,json,os,runpy,subprocess,sys,tarfile,time
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
WORKFLOW=REPO/'research/connections/stream-mlp'


def cmd(args,cwd=None):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def save(root,state):
    tmp=root/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'state.json')


def collect(root,state,ad):
    with tarfile.open(root/'results'/('gate-'+state['job']+'.tar')) as tf:
        for member in tf.getmembers():
            path=Path(member.name)
            assert not path.is_absolute() and '..' not in path.parts
            assert not member.name.endswith(('.pt','.wandb'))
            if member.isfile():
                dest=root/'collected'/path;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(tf.extractfile(member).read())
    assert ad.get('ExitCode')==0,ad
    value=json.loads((root/'collected/results/stream-mlp/validation.json').read_text())
    assert value['complete'] and not value['formal_training'] and value['source']==state['source']
    assert value['checks']==dict(tests=18,failures=0,errors=0,skipped=0)
    assert value['restart']['complete'] and value['restart']['steps']==128
    assert cmd(['git','rev-parse','HEAD'],REPO)==state['source']['commit']
    assert not cmd(['git','status','--porcelain'],REPO)
    report=WORKFLOW/'validation.json';report.write_text(json.dumps(dict(value,job=state['job']),indent=2)+'\n')
    cmd(['git','add',str(report)],REPO);cmd(['git','commit','-m','Validate shared stream MLP mixing on GPU'],REPO)
    state.update(status='validated_pending_push',reviewed_head=cmd(['git','rev-parse','HEAD'],REPO));save(root,state)


def publish(root,state):
    assert cmd(['git','rev-parse','HEAD'],REPO)==state['reviewed_head']
    assert not cmd(['git','status','--porcelain'],REPO)
    try:
        head=runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']()
    except Exception as exc:
        # A push can reach GitHub before its local PTY waiter times out.
        head=cmd(['git','rev-parse','HEAD'],REPO)
        assert cmd(['git','ls-remote','origin','refs/heads/main'],REPO).split()[0]==head
        state['publisher_wait_error']=repr(exc)
    state.update(status='published',pushed_commit=head);save(root,state)


def submit_training(root,state):
    train_root=root.parent/'stream-mlp-10b-20261105'
    subprocess.run([sys.executable,str(WORKFLOW/'submit_train.py'),str(train_root),str(root)],cwd=REPO,check=True,timeout=120)
    training=json.loads((train_root/'state.json').read_text())
    pidpath=train_root/'monitor.pid';pid=int(pidpath.read_text()) if pidpath.exists() else None;alive=False
    if pid:
        try:os.kill(pid,0);alive=True
        except ProcessLookupError:pass
    if not alive and training['status'] not in ('training_started','needs_attention'):
        with (train_root/'logs/watch-start.out').open('a') as out,(train_root/'logs/watch-start.err').open('a') as err:
            proc=subprocess.Popen([sys.executable,str(WORKFLOW/'watch_start.py'),str(train_root)],cwd=REPO,stdout=out,stderr=err,start_new_session=True)
        pid=proc.pid;pidpath.write_text(str(pid)+'\n')
    state.update(status='complete',training_job=training['job'],run_id=training['run_id'],training_monitor_pid=pid);save(root,state)
    agents=Path('/home/yzhong239/AGENTS.md')
    with agents.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);text=f.read()
        heading='## 新授权：共享4→16→4 Stream MLP mixing 无Sinkhorn，单一10B（2026-11-05）\n'
        assert heading in text
        note=f'\nGPU gate{state["job"]}已exit0，18检查、同卡profile及25→128真实恢复通过。main{state["pushed_commit"]}已push核验；正式10B{state["training_job"]}/W&B{state["run_id"]}已提交，首步watch PID{pid}/300秒。提交不等于已训练；短检查不保证长期稳定/效果。\n'
        f.seek(0);f.write(text.replace(heading,heading+note,1));f.truncate()


def main(root):
    root=Path(root).resolve()
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            state=json.loads((root/'state.json').read_text())
            if state['status'] in ('complete','needs_attention'):return
            state['monitor_pid']=os.getpid()
            try:
                if state['status']=='validated_pending_push':publish(root,state)
                if state['status']=='published':submit_training(root,state);return
                attrs='ClusterId,ProcId,JobStatus,ExitCode,RemoteHost,HoldReason'
                rows=json.loads(cmd(['condor_q',state['job'],'-json','-attributes',attrs]) or '[]')
                if not rows:rows=json.loads(cmd(['condor_history',state['job'],'-limit','1','-json','-attributes',attrs]) or '[]')
                state.update(last_job=rows[0] if rows else None,last_poll=time.time())
                if rows and rows[0]['JobStatus']==4:collect(root,state,rows[0]);continue
                if rows and rows[0]['JobStatus'] in (3,5):state['status']='needs_attention'
                save(root,state)
            except Exception as exc:
                state.update(status='needs_attention',error=repr(exc));save(root,state);return
            if state['status']=='needs_attention':return
            time.sleep(300)


if __name__=='__main__':main(sys.argv[1])
