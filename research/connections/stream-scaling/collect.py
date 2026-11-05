"""Collect a bounded optimization check, publish its report, then stop."""
import fcntl,json,os,runpy,subprocess,sys,tarfile,time
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
WORKFLOW=REPO/'research/connections/stream-scaling'


def cmd(args,cwd=None):return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def save(root,state):
    p=root/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'state.json')


def collect(root,state,ad):
    archive=root/'results'/('gate-'+state['job']+'.tar')
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            p=Path(m.name);assert not p.is_absolute() and '..' not in p.parts
            assert not m.name.endswith(('.pt','.wandb'))
            if m.isfile():
                dest=root/'collected'/p;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(tf.extractfile(m).read())
    assert ad.get('ExitCode')==0,ad
    value=json.loads((root/'collected/results/stream-scaling/validation.json').read_text())
    assert value['complete'] and not value['formal_training'] and value['source']==state['source']
    assert value['checks']==dict(tests=30,failures=0,errors=0,skipped=0)
    assert value['speed_only'] and value['parameter_updates']==0 and value['streams']==[4,8,16,32]
    assert all(r['parameter_updates']==0 for r in value['records'])
    assert cmd(['git','rev-parse','HEAD'],REPO)==state['source']['commit']
    assert not cmd(['git','status','--porcelain'],REPO)
    report=WORKFLOW/'validation.json';report.write_text(json.dumps(dict(value,job=state['job']),indent=2)+'\n')
    cmd(['git','add',str(report)],REPO);cmd(['git','commit','-m','Measure frozen mHC and stream MLP scaling through32 streams'],REPO)
    state.update(status='validated_pending_push',reviewed_head=cmd(['git','rev-parse','HEAD'],REPO));save(root,state)


def publish(root,state):
    assert cmd(['git','rev-parse','HEAD'],REPO)==state['reviewed_head']
    assert not cmd(['git','status','--porcelain'],REPO)
    try:head=runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']()
    except Exception as exc:
        head=cmd(['git','rev-parse','HEAD'],REPO)
        assert cmd(['git','ls-remote','origin','refs/heads/main'],REPO).split()[0]==head
        state['publisher_wait_error']=repr(exc)
    state.update(status='complete',pushed_commit=head,formal_training=False);save(root,state)
    report=json.loads((WORKFLOW/'validation.json').read_text())
    agents=Path('/home/yzhong239/AGENTS.md')
    with agents.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);s=f.read()
        heading='## 新授权：mHC与Stream MLP的4/8/16/32流纯速度测试（2026-11-05）\n'
        assert heading in s
        note=f'\nGPU job{state["job"]}已exit0，30检查及同卡4/8/16/32流纯测速完成，统一fullmodel batch={report["selected_common_model_batch"]}；main{head}已push核验。没有权重更新/正式训练/W&B/ckpt。通用benchmark实现与官方4流anchor分列，不声称NVIDIA官方支持8/16/32；collector已退出。\n'
        f.seek(0);f.write(s.replace(heading,heading+note,1));f.truncate()


def main(root):
    root=Path(root).resolve()
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            state=json.loads((root/'state.json').read_text())
            if state['status'] in ('complete','needs_attention'):return
            state['monitor_pid']=os.getpid()
            try:
                if state['status']=='validated_pending_push':publish(root,state);return
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
