"""Reuse moment job setup; collect/publish only, never submit formal training."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import runpy
import tarfile
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('moment_workflow',HERE.parent/'moments/workflow.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.KEY='jensen-silu-gate-20261103'


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
                    with tarfile.open(root/'results'/f'gate-{state["job"]}.tar') as tf:
                        result=json.load(tf.extractfile('results/jensen-gate.json'))
                        assert result['complete'] and result['source']==state['source']
                        assert result['checks']==dict(tests=25,failures=0,errors=0,skipped=0)
                        assert not result['formal_training'] and not result['weights_returned']
                        assert len(result['profiles'])==3
                        assert {r['mode'] for r in result['profiles']}=={'none','variance','jensen_silu'}
                        for row in result['profiles']:
                            assert row['peak_reserved_bytes']<.9*22000*1024**2 and len(row['samples_seconds'])==30
                        assert result['restart']['exact_steps_1_to_50']
                        assert result['restart']['summary']['complete'] and result['restart']['summary']['steps']==50
                        for m in tf.getmembers():
                            assert not m.name.endswith(('.pt','.wandb'))
                            if m.isfile():
                                parts=Path(m.name).parts;assert parts[0]=='results' and '..' not in parts
                                dest=root/'collected'/Path(*parts[1:]);dest.parent.mkdir(parents=True,exist_ok=True)
                                dest.write_bytes(tf.extractfile(m).read())
                    assert not base.command(['git','status','--porcelain'])
                    assert not base.command(['git','diff','--name-only',state['source']['commit'],'HEAD','--',
                        'nanochat','scripts','tests','research/connections/jensen',
                        'research/connections/runtime','research/connections/toolchain'])
                    report=HERE/'validation.json';base.write(report,dict(result,job=state['job']))
                    base.command(['git','add',str(report)])
                    base.command(['git','commit','-m','Record verified SiLU Jensen gap and recovery'])
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
