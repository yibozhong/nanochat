"""One restart gate, publish, then three new-backend 10B runs; stop at first steps."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import runpy
import tarfile
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('moment_workflow',HERE.parent.parent/'moments/workflow.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.KEY='triton-moments-10b-20261103'
base.MODES=('none','variance','jensen_silu')
VALIDATED='bc77aba06ec52e1e284123340b5bd3e414a815b1'
PRIOR_GATE=Path('/home/yzhong239/triton-moments-gate-20261103/retry-2')
PRODUCTION=['nanochat','scripts/connection_train.py','scripts/connection_experiment.py',
            'research/connections/runtime','research/connections/toolchain']


def validated_production(commit):
    prior=json.loads((PRIOR_GATE/'state.json').read_text())
    assert prior['status']=='complete' and prior['source']['commit']==VALIDATED
    assert json.loads((PRIOR_GATE/'published.json').read_text())['remote_verified']
    assert not base.command(['git','diff','--name-only',VALIDATED,commit,'--',*PRODUCTION])


def submit(root):
    validated_production('HEAD')
    base.submit(root,HERE)


def audit_publish(root,state):
    validated_production(state['source']['commit'])
    with tarfile.open(root/'results'/f'gate-{state["job"]}.tar') as tf:
        result=json.load(tf.extractfile('results/triton-resume-gate.json'))
        assert result['complete'] and result['source']==state['source'] and not result['formal_training']
        assert result['checks']==dict(tests=4,failures=0,errors=0,skipped=0)
        assert set(result['trials'])==set(base.MODES)
        for mode,row in result['trials'].items():
            summary,config=row['summary'],row['config']
            assert row['exact_steps_1_to_50'] and summary['complete']
            assert summary['steps']==50 and summary['training_tokens']==1638400
            assert config['model']['attention_kernel']=='fla_triton'
            assert config['model']['attention_statistic']==mode
            assert config['logging_backend']=='local' and config['protocol']['schedule_steps']==305176
        for member in tf.getmembers():
            assert not member.name.endswith(('.pt','.wandb'))
            if member.isfile():
                parts=Path(member.name).parts
                assert parts[0]=='results' and '..' not in parts and not Path(member.name).is_absolute()
                dest=root/'collected'/Path(*parts[1:]);dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(tf.extractfile(member).read())
    # Reuse the completed, identical-production-source performance gate.
    profiles=json.loads((HERE.parent/'numerical-audit.json').read_text())['full_step_summary']
    summary={mode:dict(mean_seconds=profiles[mode]['step_ms']/1000,
        peak_allocated_bytes=round(profiles[mode]['allocated_gb']*1e9),
        peak_reserved_bytes=round(profiles[mode]['reserved_gb']*1e9),
        parameters=58720256 if mode=='none' else 60817408) for mode in base.MODES}
    summary['baseline']=summary['none']  # Existing training harness lookup for mode=none.
    result.update(job=state['job'],kernel_gate='7802199.0',kernel_source=VALIDATED,profile_reference=summary)
    base.write(root/'validation.json',result)
    assert not base.command(['git','status','--porcelain'])
    assert not base.command(['git','diff','--name-only',state['source']['commit'],'HEAD','--',
        *PRODUCTION,'scripts/connection_triton_resume_gate.py','tests/test_triton_resume.py',str(HERE.relative_to(base.REPO))])
    report=HERE/'recovery-validation.json';base.write(report,result)
    base.command(['git','add',str(report)])
    base.command(['git','commit','-m','Validate Triton attention checkpoint recovery for three 10B runs'])
    head=base.command(['git','rev-parse','HEAD'])
    try:runpy.run_path(str(base.REPO/'research/connections/nvidia-only/watch.py'))['push']()
    except Exception:
        assert base.command(['git','ls-remote','origin','refs/heads/main']).split()[0]==head
    assert base.command(['git','ls-remote','origin','refs/heads/main']).split()[0]==head
    base.write(root/'published.json',dict(commit=head,gate_commit=state['source']['commit'],job=state['job'],remote_verified=True))
    return summary


def monitor(root):
    # Keep the same lock during gate collection; release before base.monitor reacquires it.
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            state=json.loads((root/'state.json').read_text())
            if state['status'] in ('training_started','needs_attention','stopped_by_user'):return
            if state.get('training_jobs'):break
            try:
                info=base.job_info(state['job']);state['last_gate']=info
                if info and info['JobStatus']==4:
                    assert info.get('ExitCode')==0,info
                    summary=audit_publish(root,state)
                    base.submit_training(root,state,summary,HERE)
                    break
                if info and info['JobStatus'] in (3,5):raise RuntimeError(str(info))
                state['last_poll']=time.time();base.write(root/'state.json',state)
            except Exception as exc:
                state.update(status='needs_attention',error=repr(exc));base.write(root/'state.json',state);raise
            time.sleep(30)
        else:return
    base.monitor(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('action',choices=('submit','monitor'))
    args=parser.parse_args();globals()[args.action](args.root.resolve())
