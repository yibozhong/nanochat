"""Absolute Jensen diagnostics only: collect, audit, publish; no formal training."""
import argparse
import fcntl
import importlib.util
import json
import math
from pathlib import Path
import runpy
import tarfile
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('moment_workflow',HERE.parent/'moments/workflow.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.KEY='jensen-abs-gate-20261103'


def audit(root,state):
    with tarfile.open(root/'results'/f'gate-{state["job"]}.tar') as tf:
        result=json.load(tf.extractfile('results/abs-jensen-gate.json'))
        assert result['complete'] and result['source']==state['source']
        assert result['checks']==dict(tests=28,failures=0,errors=0,skipped=0)
        assert not result['formal_training'] and not result['weights_returned']
        assert len(result['profiles'])==6 and len({r['gpu'] for r in result['profiles']})==1
        assert {r['mode'] for r in result['profiles']}=={'none','jensen_silu','jensen_abs'}
        for row in result['profiles']:
            assert row['attention_kernel']=='fla_triton'
            assert row['peak_reserved_bytes']<.9*22000*1024**2 and len(row['samples_seconds'])==30
            assert all(math.isfinite(x) and x>0 for x in row['samples_seconds'])
            assert row['parameters']==(58720256 if row['mode']=='none' else 60817408)
        restart=result['restart']
        assert restart['exact_steps_1_to_50'] and restart['summary']['complete'] and restart['summary']['steps']==50
        assert restart['config']['protocol']['schedule_steps']==305176 and restart['config']['logging_backend']=='local'
        result['operator_profile']=json.load(tf.extractfile('results/operator-profile.json'))
        for row in result['operator_profile']:
            counts=row['kernels'];assert len(counts)==4 and set(counts.values())=={1},counts
        numeric={}
        for member in tf.getmembers():
            assert not member.name.endswith(('.pt','.wandb'))
            if member.isfile():
                parts=Path(member.name).parts
                assert parts[0]=='results' and '..' not in parts and not Path(member.name).is_absolute()
                data=tf.extractfile(member).read()
                dest=root/'collected'/Path(*parts[1:]);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
                if member.name.startswith('results/triton-numerics/'):
                    numeric[Path(member.name).stem]=json.loads(data)
        result['numerics']=numeric
    summary={}
    for mode in ('none','jensen_silu','jensen_abs'):
        rows=[r for r in result['profiles'] if r['mode']==mode];assert len(rows)==2
        summary[mode]=dict(mean_seconds=sum(r['mean_seconds'] for r in rows)/2,
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows))
    result.update(job=state['job'],summary=summary)
    return result


def monitor(root):
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            state=json.loads((root/'state.json').read_text())
            if state['status'] in ('complete','needs_attention','stopped_by_user'):return
            try:
                info=base.job_info(state['job']);state['last_job']=info
                if info and info['JobStatus']==4:
                    assert info.get('ExitCode')==0,info
                    result=audit(root,state)
                    assert not base.command(['git','status','--porcelain'])
                    assert not base.command(['git','diff','--name-only',state['source']['commit'],'HEAD','--',
                        'nanochat','scripts','tests','research/connections/jensen-abs',
                        'research/connections/runtime','research/connections/toolchain'])
                    report=HERE/'validation.json';base.write(report,result)
                    base.command(['git','add',str(report)])
                    base.command(['git','commit','-m','Validate fused absolute Jensen attention and restart'])
                    head=base.command(['git','rev-parse','HEAD'])
                    try:runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']()
                    except Exception:
                        assert base.command(['git','ls-remote','origin','refs/heads/main']).split()[0]==head
                    assert base.command(['git','ls-remote','origin','refs/heads/main']).split()[0]==head
                    base.write(root/'published.json',dict(commit=head,gate_commit=state['source']['commit'],remote_verified=True,job=state['job']))
                    state.update(status='complete',formal_training=False)
                elif info and info['JobStatus'] in (3,5):raise RuntimeError(str(info))
                state['last_poll']=time.time();base.write(root/'state.json',state)
            except Exception as exc:
                state.update(status='needs_attention',error=repr(exc));base.write(root/'state.json',state);raise
            if state['status']=='complete':return
            time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('action',choices=('submit','monitor'))
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='submit':base.submit(root,experiment_dir=HERE)
    else:monitor(root)
