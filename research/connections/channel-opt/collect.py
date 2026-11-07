"""Collect and publish a bounded optimization result. Never launches training."""
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

REPO=Path(__file__).resolve().parents[3]


def cmd(args,cwd=None):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def save(root,state):
    p=root/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'state.json')


def validate(v,state):
    assert v['complete'] and not v['formal_training'] and not v['new_formal_run'] and not v['default_changed']
    assert v['source']==state['source'] and v['checks']==dict(tests=47,failures=0,errors=0,skipped=0)
    assert v['restart']['complete'] and v['restart']['steps']==128
    assert all(math.isfinite(v['restart'][k]) for k in ('initial_dev_ce','final_dev_ce'))
    assert v['restart']['final_dev_ce']<v['restart']['initial_dev_ce']
    names={'baseline',*[f'{k}-compact{cp}' for k in ('legacy','split','fused','fused_wide') for cp in (0,1)],*[f'abba-{i}' for i in range(4)]}
    assert set(v['profiles'])==names
    for p in v['profiles'].values():
        assert p['complete'] and p['status']=='fits' and p['optimizer_updates']==60 and p['causality_passed']
        assert p['source']==state['source'] and p['microbatch']==16 and p['batch_tokens']==32768
        assert 'H200' in p['gpu'] and p['actual_attention_backend']=='fa3'
        assert len(p['losses'])==60 and all(math.isfinite(x) for x in p['losses'])
        assert len(p['step_seconds'])==40 and all(math.isfinite(x) and x>0 for x in p['step_seconds'])
    assert len({p['gpu_uuid'] for p in v['profiles'].values()})==1
    assert len({p['input_sha256'] for p in v['profiles'].values()})==1
    assert v['selected_kernel'] in ('legacy','split','fused','fused_wide')
    micro=v['micro'];assert micro['complete'] and micro['optimizer_updates']==0
    assert micro['source']==state['source'] and micro['gpu_uuid']==v['profiles']['baseline']['gpu_uuid']
    assert set(micro['results'])=={'legacy','split','fused','fused_wide'}
    for name,rows in micro['results'].items():
        assert len(rows)==2 and micro['traces'][name]['kernel_count']>0
        for row in rows:
            assert math.isfinite(row['mean_seconds']) and row['mean_seconds']>0
            assert row['errors']['output']<.006 and max(row['errors']['gradients'])<.012
    old=(v['profiles']['abba-0']['mean_seconds']+v['profiles']['abba-3']['mean_seconds'])/2
    new=(v['profiles']['abba-1']['mean_seconds']+v['profiles']['abba-2']['mean_seconds'])/2
    assert old==v['old_seconds'] and new==v['new_seconds']
    assert v['target_reached']==(new<=.09)
    assert v['recommended']==(new<old*.98)


def existing_unchanged(root):
    for name,digest in json.loads((root/'existing-training-inputs.json').read_text()).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name


def collect(root,state,ad):
    with tarfile.open(root/'results'/('gate-'+state['job']+'.tar')) as tf:
        for m in tf.getmembers():
            p=Path(m.name)
            assert not p.is_absolute() and '..' not in p.parts and not m.issym() and not m.islnk()
            assert not m.name.endswith(('.pt','.wandb'))
            if m.isfile():
                out=root/'collected'/p;out.parent.mkdir(parents=True,exist_ok=True)
                out.write_bytes(tf.extractfile(m).read())
    assert ad.get('ExitCode')==0,ad
    value=json.loads((root/'collected/results/channel-fast/validation.json').read_text())
    validate(value,state);existing_unchanged(root)
    assert cmd(['git','rev-parse','HEAD'],REPO)==state['source']['commit']
    assert not cmd(['git','status','--porcelain'],REPO)
    p=REPO/'research/connections/channel-opt/validation.json'
    p.write_text(json.dumps(dict(value,job=state['job']),indent=2)+'\n')
    cmd(['git','add',str(p)],REPO);cmd(['git','commit','-m','Validate and profile channelwise kernel candidates on H200'],REPO)
    state.update(status='validated_pending_push',reviewed_head=cmd(['git','rev-parse','HEAD'],REPO),
        selected_kernel=value['selected_kernel'],compact_routing=value['selected_compact_routing'],
        new_ms=1000*value['new_seconds'],old_ms=1000*value['old_seconds'],target_reached=value['target_reached'])
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
                    existing_unchanged(root)
                    assert cmd(['git','rev-parse','HEAD'],REPO)==state['reviewed_head']
                    assert not cmd(['git','status','--porcelain'],REPO)
                    state['pushed_commit']=runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']()
                    state.update(status='complete',new_formal_run=False);save(root,state)
                    p=Path('/home/yzhong239/AGENTS.md')
                    with p.open('r+') as f:
                        fcntl.flock(f,fcntl.LOCK_EX);s=f.read();h='## 新授权：Channelwise kernel优化与90ms目标（2026-11-06）\n'
                        assert h in s
                        note=f'\nGate{state["job"]}已exit0，47检查、生产形状CUDA trace、13组60步profile及25→128恢复通过。选择{state["selected_kernel"]}/compact={state["compact_routing"]}，ABBA旧{state["old_ms"]:.3f}ms/新{state["new_ms"]:.3f}ms，90ms目标达成={state["target_reached"]}。main{state["pushed_commit"]}已push；默认legacy保留，当前两正式训练inputs不变，无新正式run。collector已退出。\n'
                        f.seek(0);f.write(s.replace(h,h+note,1));f.truncate()
                    return
                attrs='ClusterId,ProcId,JobStatus,ExitCode,HoldReason,RemoteHost'
                rows=json.loads(cmd(['condor_q',state['job'],'-json','-attributes',attrs]) or '[]')
                if not rows:rows=json.loads(cmd(['condor_history',state['job'],'-limit','1','-json','-attributes',attrs]) or '[]')
                state.update(last_job=rows[0] if rows else None,last_poll=time.time())
                if rows and rows[0]['JobStatus']==4:collect(root,state,rows[0])
                elif rows and rows[0]['JobStatus'] in (3,5):state['status']='needs_attention'
                save(root,state)
            except Exception as exc:
                state.update(status='needs_attention',error=repr(exc));save(root,state);return
            if state['status']=='needs_attention':return
            if state['status']=='validated_pending_push':continue
            time.sleep(300)


if __name__=='__main__':main(sys.argv[1])
