"""Validate retained statistics and publish cleanup; never launch training."""
import argparse,fcntl,importlib.util,json,runpy,tarfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('moment_workflow',HERE.parent/'moments/workflow.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.KEY='attention-statistics-cleanup-20261104'


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
                        result=json.load(tf.extractfile('results/cleanup-gate.json'))
                        assert result['complete'] and result['source']==state['source']
                        assert result['checks']==dict(tests=35,failures=0,errors=0,skipped=0)
                        assert not result['formal_training'] and not result['weights_returned']
                        assert result['supported_statistics']==['none','variance','jensen_silu']
                        numeric={}
                        for member in tf.getmembers():
                            assert not member.name.endswith(('.pt','.wandb'))
                            if not member.isfile():continue
                            parts=Path(member.name).parts
                            assert parts[0]=='results' and '..' not in parts and not Path(member.name).is_absolute()
                            data=tf.extractfile(member).read()
                            dest=root/'collected'/Path(*parts[1:]);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
                            if member.name.startswith('results/triton-numerics/'):numeric[Path(member.name).stem]=json.loads(data)
                    result.update(job=state['job'],numerics=numeric)
                    assert not base.command(['git','status','--porcelain'])
                    assert not base.command(['git','diff','--name-only',state['source']['commit'],'HEAD','--',
                        'nanochat','scripts','tests','research/connections/statistics-cleanup',
                        'research/connections/runtime','research/connections/toolchain'])
                    report=HERE/'validation.json';base.write(report,result)
                    base.command(['git','add',str(report)])
                    base.command(['git','commit','-m','Validate retained variance and SiLU Jensen after statistics cleanup'])
                    head=base.command(['git','rev-parse','HEAD'])
                    try:runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']()
                    except Exception:
                        assert base.command(['git','ls-remote','origin','refs/heads/main']).split()[0]==head
                    assert base.command(['git','ls-remote','origin','refs/heads/main']).split()[0]==head
                    base.write(root/'published.json',dict(commit=head,gate_commit=state['source']['commit'],job=state['job'],remote_verified=True))
                    state.update(status='complete',formal_training=False,automatic_continuation=False)
                elif info and info['JobStatus'] in (3,5):raise RuntimeError(str(info))
                state['last_poll']=time.time();base.write(root/'state.json',state)
            except Exception as exc:
                state.update(status='needs_attention',error=repr(exc));base.write(root/'state.json',state);raise
            if state['status']=='complete':return
            time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('action',choices=('submit','monitor'))
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='submit':base.submit(root,HERE)
    else:monitor(root)
