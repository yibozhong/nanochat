"""Validate results, publish, and submit exactly one authorized attnres-scale10B."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import time
from validation import validate

REPO=Path(__file__).resolve().parents[3]


def command(args,cwd=REPO):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def save(root,state):
    tmp=root/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'state.json')


def check_retained(root):
    old=json.loads((root/'retained-training.json').read_text())
    for files in old.values():
        for name,expected in files.items():
            p=Path(name);st=p.stat()
            assert dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)==expected,name
    for name,expected in json.loads((root/'retained-attnres-stat.json').read_text()).items():
        st=Path(name).stat()
        assert dict(size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)==expected,name


def collect(root,state,ad):
    archive=root/'results'/('gate-'+state['job']+'.tar')
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            name=Path(m.name)
            assert not name.is_absolute() and '..' not in name.parts and not m.issym() and not m.islnk()
            assert not m.name.endswith(('.pt','.wandb'))
            if m.isfile():
                dest=root/'collected'/name;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(tf.extractfile(m).read())
    assert ad.get('ExitCode')==0,ad
    v=json.loads((root/'collected/results/attnres-scale/validation.json').read_text())
    validate(v,state['source']);check_retained(root)
    assert command(['git','rev-parse','HEAD'])==state['reviewed_head']
    assert not command(['git','status','--porcelain'])
    # Workflow-only additions after the immutable GPU snapshot are permitted;
    # the complete model/runtime/training/test execution tree must match.
    from submit_train import execution_files
    execution=execution_files(state['source']['commit'])
    v.update(job=state['job'],archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    report=REPO/'research/connections/attnres-scale/validation.json'
    report.write_text(json.dumps(v,indent=2)+'\n')
    audit=report.with_name('gpu-source-audit.json')
    audit.write_text(json.dumps(dict(gate_source=state['source']['commit'],files=execution,retained_training_unchanged=True),indent=2)+'\n')
    (root/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
    command(['git','add',str(report),str(audit)])
    command(['git','commit','-m','Validate static channel AttnRes values and matched GPU throughput'])
    state.update(status='validated_pending_push',reviewed_head=command(['git','rev-parse','HEAD']));save(root,state)


def update_memory(state,training):
    path=Path('/home/yzhong239/AGENTS.md')
    heading='## 最新授权：停止并清理QK-scale，静态Channel AttnRes验证后单一10B（2026-11-08）\n'
    note=f'\nGPU gate{state["job"]}已exit0，24检查/同卡ABCCBA60步/25→128恢复通过，main{state["pushed_commit"]}已push核验。唯一正式{training["job"]}/W&B{training["run_id"]}已held审计后release；调度与首步观察PID{state["training_monitor_pid"]}每60秒，连续Idle3600秒后才放开A100/PRO6000，运行不迁移。提交不等于正式开跑；新GPU family先24检查+25→60再正式初始化/恢复。\n'
    with path.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);text=f.read();assert heading in text
        f.seek(0);f.write(text.replace(heading,heading+note,1));f.truncate()


def main(root):
    root=Path(root).resolve()
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not any((root/n).exists() for n in ('STOPPED','STOPPED.json')):
            state=json.loads((root/'state.json').read_text())
            if state['status'] in ('complete','needs_attention'):return
            state['monitor_pid']=os.getpid()
            try:
                if state['status']=='validated_pending_push':
                    assert command(['git','rev-parse','HEAD'])==state['reviewed_head']
                    assert not command(['git','status','--porcelain'])
                    state['pushed_commit']=runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']()
                    state['status']='published';save(root,state)
                if state['status']=='published':
                    assert state['automatic_training'] and not any((root/n).exists() for n in ('STOPPED','STOPPED.json'))
                    train_root=root.parent/'attnres-scale-10b-20261108'
                    subprocess.run([sys.executable,str(REPO/'research/connections/attnres-scale/submit_train.py'),str(train_root)],check=True,cwd=REPO,timeout=180)
                    training=json.loads((train_root/'state.json').read_text())
                    assert (train_root/'submission-verified.json').exists()
                    with (train_root/'logs/watch.out').open('a') as out,(train_root/'logs/watch.err').open('a') as err:
                        proc=subprocess.Popen([sys.executable,str(REPO/'research/connections/attnres-scale/watch_train.py'),str(train_root)],cwd=REPO,stdout=out,stderr=err,start_new_session=True)
                    (train_root/'monitor.pid').write_text(str(proc.pid)+'\n')
                    state.update(status='complete',training_job=training['job'],run_id=training['run_id'],training_monitor_pid=proc.pid)
                    save(root,state);update_memory(state,training);return
                attrs='ClusterId,ProcId,JobStatus,ExitCode,RemoteHost,HoldReason,NumJobStarts'
                rows=json.loads(command(['condor_q',state['job'],'-json','-attributes',attrs]) or '[]')
                if not rows:rows=json.loads(command(['condor_history',state['job'],'-limit','1','-json','-attributes',attrs]) or '[]')
                state.update(last_job=rows[0] if rows else None,last_poll=time.time())
                if rows and rows[0]['JobStatus']==4:collect(root,state,rows[0])
                elif rows and rows[0]['JobStatus'] in (3,5):state['status']='needs_attention'
                save(root,state)
            except Exception as exc:
                state.update(status='needs_attention',error=repr(exc));save(root,state);print(repr(exc),flush=True);return
            if state['status']=='needs_attention':return
            if state['status'] in ('validated_pending_push','published'):continue
            time.sleep(300)


if __name__=='__main__':main(sys.argv[1])
