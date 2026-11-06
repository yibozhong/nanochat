"""Collect the GPU gate, publish, launch exactly one authorized10B, then stop."""
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

REPO=Path(__file__).resolve().parents[3]


def cmd(args,cwd=None):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def save(root,state):
    tmp=root/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'state.json')


def validate(value,state):
    assert value['complete'] and not value['formal_training'] and not value['preflight']
    assert value['source']==state['source']
    assert value['checks']==dict(tests=25,failures=0,errors=0,skipped=0)
    assert value['restart']['complete'] and value['restart']['steps']==128
    assert all(math.isfinite(value['restart'][k]) for k in ('initial_dev_ce','final_dev_ce'))
    assert value['restart']['final_dev_ce']<value['restart']['initial_dev_ce']
    assert value['microbatch'] in (16,8,4)
    assert set(value['profiles'])=={'baseline','mhc_no_sinkhorn','mhc_channel_no_sinkhorn','gated_hc'}
    for variant,p in value['profiles'].items():
        assert p['variant']==variant and p['microbatch']==value['microbatch']
        assert p['source']==state['source'] and p['batch_tokens']==32768 and p['sequence_length']==2048
        assert p['complete'] and p['status']=='fits' and p['optimizer_updates']==60
        assert p['causality_passed'] and math.isfinite(p['mean_seconds']) and p['mean_seconds']>0
        assert len(p['losses'])==60 and all(math.isfinite(loss) for loss in p['losses'])
        assert len(p['step_seconds'])==40 and all(math.isfinite(t) and t>0 for t in p['step_seconds'])
        assert p['peak_reserved_bytes']<.9*22000*1024**2
    assert len({p['gpu_uuid'] for p in value['profiles'].values()})==1
    assert len({p['input_sha256'] for p in value['profiles'].values()})==1
    assert value['profiles']['gated_hc']['parameters']==59679157
    micro=value['micro'];assert micro['complete'] and micro['shape']==[value['microbatch'],2048,512,4]
    assert micro['gpu_uuid']==value['profiles']['gated_hc']['gpu_uuid']
    assert set(micro['results'])=={'fused','compiled_torch'}
    for r in micro['results'].values():
        assert math.isfinite(r['mean_seconds']) and r['mean_seconds']>0


def collect(root,state,ad):
    archive=root/'results'/('gate-'+state['job']+'.tar')
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            path=Path(m.name)
            assert not path.is_absolute() and '..' not in path.parts
            assert not m.name.endswith(('.pt','.wandb')) and not m.issym() and not m.islnk()
            if m.isfile():
                dest=root/'collected'/path;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(tf.extractfile(m).read())
    assert ad.get('ExitCode')==0,ad
    value=json.loads((root/'collected/results/gated-hc/validation.json').read_text())
    validate(value,state)
    assert cmd(['git','rev-parse','HEAD'],REPO)==state['source']['commit']
    assert not cmd(['git','status','--porcelain'],REPO)
    report=REPO/'research/connections/gated-hc/validation.json'
    report.write_text(json.dumps(dict(value,job=state['job']),indent=2)+'\n')
    cmd(['git','add',str(report)],REPO)
    cmd(['git','commit','-m','Validate SiLU gated HC on GPU'],REPO)
    state.update(status='validated_pending_push',reviewed_head=cmd(['git','rev-parse','HEAD'],REPO))
    save(root,state)


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
                    state['pushed_commit']=runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']()
                    state['status']='published';save(root,state)
                if state['status']=='published':
                    assert not (root/'STOPPED.json').exists()
                    train_root=root.parent/'gated-hc-10b-20261106'
                    subprocess.run([sys.executable,str(REPO/'research/connections/gated-hc/submit_train.py'),str(train_root)],check=True,cwd=REPO,timeout=180)
                    training=json.loads((train_root/'state.json').read_text())
                    # One watcher only, no second formal submission on restart.
                    with (train_root/'logs/watch-start.out').open('a') as out,(train_root/'logs/watch-start.err').open('a') as err:
                        proc=subprocess.Popen([sys.executable,str(REPO/'research/connections/gated-hc/watch_start.py'),str(train_root)],cwd=REPO,stdout=out,stderr=err,start_new_session=True)
                    (train_root/'monitor.pid').write_text(str(proc.pid)+'\n')
                    state.update(status='complete',training_job=training['job'],run_id=training['run_id'],training_monitor_pid=proc.pid)
                    save(root,state)
                    agents=Path('/home/yzhong239/AGENTS.md')
                    with agents.open('r+') as f:
                        fcntl.flock(f,fcntl.LOCK_EX)
                        text=f.read();heading='## 新授权：SiLU Gated HC，验证后单一10B（2026-11-06）\n'
                        assert heading in text
                        note=f'\nGPU gate{state["job"]}已exit0，25检查/同卡60步profile/25→128真实恢复通过。main{state["pushed_commit"]}已push核验。正式10B{state["training_job"]}/W&B{state["run_id"]}已提交，首步watch PID{state["training_monitor_pid"]}/300秒；提交不等于训练开始，短测不保证长期稳定或收益。\n'
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
            if state['status'] in ('validated_pending_push','published'):continue
            time.sleep(300)


if __name__=='__main__':main(sys.argv[1])
