"""Launch exactly the authorized FA3 looping 10B; monitor until first finite step."""
import argparse,fcntl,hashlib,importlib.util,json,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
GATE=Path('/home/yzhong239/attention-looping-fa3-20261104')
spec=importlib.util.spec_from_file_location('moment_workflow',HERE.parent/'moments/workflow.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.KEY='attention-looping-10b-20261104'
base.MODES=('attention_looping',)


def start(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/'STOPPED.json').exists()
        if (root/'state.json').exists():
            state=json.loads((root/'state.json').read_text())
            assert state.get('training_jobs'),'Recover incomplete launch; do not duplicate submission'
            print(json.dumps(state['training_jobs']));return
        assert not (root/'launch-intent.json').exists()
        assert not base.command(['git','status','--porcelain'])
        assert not json.loads(base.command(['condor_q','yzhong239','-constraint',f'MomentKey == "{base.KEY}"','-json']) or '[]')
        commit=base.command(['git','rev-parse','HEAD'])
        assert base.command(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit
        result=json.loads((GATE/'validation.json').read_text())
        assert result['checks']==dict(tests=41,failures=0,errors=0,skipped=0)
        assert result['resume']['summary']['complete'] and result['resume']['exact_steps_1_to_50']
        history=json.loads(base.command(['condor_history',result['job'],'-limit','1','-scanlimit','100000','-json','-attributes','JobStatus,ExitCode']))
        assert history[0]['JobStatus']==4 and history[0]['ExitCode']==0
        audit=json.loads((HERE/'ns-removal-audit.json').read_text())
        assert audit['unchanged_execution'] and audit['gate_commit']==result['source']['commit']
        for path,digest in audit['final_runtime_sha256'].items():
            assert hashlib.sha256((REPO/path).read_bytes()).hexdigest()==digest,path
        expected=json.loads((HERE/'fa3-artifacts.json').read_text())
        assert expected==json.loads((GATE/'collected/fa3-artifacts.json').read_text())
        summary=result['summary']
        for row in summary.values():
            row['additional_peak_reserved_bytes']=row['peak_reserved_bytes']-summary['baseline']['peak_reserved_bytes']
        for name in ('logs','results','bundles'):(root/name).mkdir(exist_ok=True)
        bundle=root/'bundles'
        (bundle/base.WHEEL).symlink_to(GATE/'bundles'/base.WHEEL)
        base.command(['git','archive','--format=tar','--prefix=nanochat-src/',f'--output={bundle/"source.tar"}',commit])
        hashes={k:v for k,v in result['source']['hashes'].items() if k not in ('source.tar','gate-data.tar')}
        hashes['source.tar']=hashlib.sha256((bundle/'source.tar').read_bytes()).hexdigest()
        source=dict(commit=commit,hashes=hashes,formal_training=False,
                    purpose='Validated FA3 attention looping after removal of inactive NS code')
        base.inputs(bundle,source)
        base.write(root/'launch-intent.json',dict(commit=commit,time=time.time(),training_runs=1,steps=305176,tokens=10000007168))
        base.write(root/'published.json',dict(commit=commit,gate_commit=result['source']['commit'],job=result['job'],remote_verified=True,ns_removed=True))
        state=dict(status='validated_for_training',job=result['job'],source=source,gate_source=result['source'],ns_removal_audit=audit)
        base.write(root/'state.json',state)
        base.submit_training(root,state,summary,HERE)
        print(json.dumps(state['training_jobs'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('action',choices=('start','monitor'))
    a=p.parse_args();root=a.root.resolve()
    if a.action=='start':start(root)
    else:
        assert json.loads((root/'state.json').read_text()).get('training_jobs')
        base.monitor(root)
