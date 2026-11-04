"""Submit and collect bounded FA3 loop validation; never submit formal training."""
import argparse,fcntl,importlib.util,json,math,tarfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('moment_workflow',HERE.parent/'moments/workflow.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.KEY='attention-looping-fa3-20261104'


def collect(root,state):
    with tarfile.open(root/'results'/f'gate-{state["job"]}.tar') as tf:
        result=json.load(tf.extractfile('results/gate.json'))
        assert result['complete'] and result['source']==state['source']
        assert result['checks']==dict(tests=41,failures=0,errors=0,skipped=0)
        assert not result['formal_training'] and not result['weights_returned']
        assert len(result['profiles'])==4 and len({p['gpu'] for p in result['profiles']})==1
        for p in result['profiles']:
            assert p['attention_backend']=='fa3' and p['attention_kernel']=='original'
            assert p['parameters']==58720256 and p['microbatch']==16 and p['grad_accum']==1
            assert len(p['samples_seconds'])==30 and all(math.isfinite(x) and x>0 for x in p['samples_seconds'])
            assert p['peak_reserved_bytes']<.9*22000*1024**2
            assert p['dev_tokens']==131072 and math.isfinite(p['dev_ce'])
        resume=result['resume'];assert resume['exact_steps_1_to_50'] and resume['summary']['complete']
        assert resume['summary']['steps']==50 and resume['config']['logging_backend']=='local'
        assert resume['config']['model']['attention_loops']==3 and resume['config']['model']['attention_loop_window']==128
        probe=json.load(tf.extractfile('results/backend-probe.json'))
        assert probe['backend']=='fa3' and probe['revision']=='de87b9b5af06dd9984df595bef90b2eba44b181a'
        for member in tf.getmembers():
            assert not member.name.endswith(('.pt','.wandb'))
            if not member.isfile():continue
            parts=Path(member.name).parts
            assert parts[0]=='results' and '..' not in parts and not Path(member.name).is_absolute()
            dest=root/'collected'/Path(*parts[1:]);dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(tf.extractfile(member).read())
    summary={}
    for mode in ('baseline','attention_looping'):
        rows=[r for r in result['profiles'] if r['mode']==mode];assert len(rows)==2
        summary[mode]=dict(mean_seconds=sum(r['mean_seconds'] for r in rows)/2,
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows),parameters=rows[0]['parameters'])
    for row in summary.values():
        row['time_relative_to_baseline']=row['mean_seconds']/summary['baseline']['mean_seconds']
        row['additional_peak_allocated_bytes']=row['peak_allocated_bytes']-summary['baseline']['peak_allocated_bytes']
    result.update(job=state['job'],summary=summary,backend_probe=probe)
    base.write(root/'validation.json',result)


def monitor(root):
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            state=json.loads((root/'state.json').read_text())
            if state['status'] in ('validated','needs_attention','stopped_by_user'):return
            try:
                info=base.job_info(state['job']);state['last_job']=info
                if info and info['JobStatus']==4:
                    assert info.get('ExitCode')==0,info
                    collect(root,state)
                    state.update(status='validated',formal_training=False,automatic_continuation=False,
                        publication='Manual audit: earlier NS commit on main is still under numerical investigation')
                elif info and info['JobStatus'] in (3,5):raise RuntimeError(str(info))
                state['last_poll']=time.time();base.write(root/'state.json',state)
            except Exception as exc:
                state.update(status='needs_attention',error=repr(exc));base.write(root/'state.json',state);raise
            if state['status']=='validated':return
            time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('action',choices=('submit','monitor'))
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='submit':base.submit(root,HERE)
    else:monitor(root)
