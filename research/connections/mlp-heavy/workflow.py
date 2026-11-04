"""GPU gate -> audited publication -> one authorized MLP-heavy 10B training."""
import argparse,importlib.util,json,math,runpy,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('moment_workflow',HERE.parent/'moments/workflow.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.KEY='mlp-heavy-10b-20261104'
base.MODES=('mlp_heavy',)


def audit_publish(root,state):
    with tarfile.open(root/'results'/f'gate-{state["job"]}.tar') as tf:
        result=json.load(tf.extractfile('results/gate.json'))
        assert result['complete'] and result['source']==state['source']
        assert result['checks']==dict(tests=37,failures=0,errors=0,skipped=0)
        assert not result['formal_training'] and not result['weights_returned']
        assert len(result['profiles'])==4 and len({p['gpu'] for p in result['profiles']})==1
        for p in result['profiles']:
            assert p['attention_backend']=='fa3' and p['attention_kernel']=='original'
            assert p['mode'] in ('baseline','mlp_heavy')
            assert p['parameters']==(58720256 if p['mode']=='baseline' else 62914560)
            assert len(p['samples_seconds'])==30 and all(math.isfinite(x) and x>0 for x in p['samples_seconds'])
            assert p['peak_reserved_bytes']<.9*22000*1024**2 and math.isfinite(p['dev_ce'])
        resume=result['resume'];assert resume['exact_steps_1_to_50'] and resume['summary']['complete']
        assert resume['summary']['steps']==50 and resume['config']['logging_backend']=='local'
        assert resume['config']['model']['mlp_heavy']
        assert resume['config']['parameters']==62914560 and resume['config']['protocol']['schedule_steps']==305176
        probe=json.load(tf.extractfile('results/backend-probe.json'))
        assert probe['backend']=='fa3' and probe['revision']=='de87b9b5af06dd9984df595bef90b2eba44b181a'
        artifacts=json.load(tf.extractfile('results/fa3-artifacts.json'))
        assert artifacts==json.loads((HERE.parent/'runtime/fa3-artifacts.json').read_text())
        for member in tf.getmembers():
            assert not member.name.endswith(('.pt','.wandb'))
            if not member.isfile():continue
            path=Path(member.name);assert not path.is_absolute() and path.parts[0]=='results' and '..' not in path.parts
            dest=root/'collected'/Path(*path.parts[1:]);dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(tf.extractfile(member).read())
    summary={}
    for mode in ('baseline','mlp_heavy'):
        rows=[p for p in result['profiles'] if p['mode']==mode];assert len(rows)==2
        summary[mode]=dict(mean_seconds=sum(p['mean_seconds'] for p in rows)/2,
            peak_allocated_bytes=max(p['peak_allocated_bytes'] for p in rows),
            peak_reserved_bytes=max(p['peak_reserved_bytes'] for p in rows),parameters=rows[0]['parameters'])
    for row in summary.values():
        row['time_relative_to_baseline']=row['mean_seconds']/summary['baseline']['mean_seconds']
        for key in ('allocated','reserved'):
            row[f'additional_peak_{key}_bytes']=row[f'peak_{key}_bytes']-summary['baseline'][f'peak_{key}_bytes']
    result.update(job=state['job'],summary=summary,backend_probe=probe,fa3_artifacts=artifacts,
                  comparison_note='No added formal baseline. Existing 10B baseline uses FLA, so final quality is not a strict matched-backend ablation. Parameter count increases 7.14%; analytic matrix FLOPs approximately match at 2K/D512.')
    base.write(root/'validation.json',result)
    assert not base.command(['git','status','--porcelain'])
    assert not base.command(['git','diff','--name-only',state['source']['commit'],'HEAD','--',
        'nanochat','scripts','tests','research/connections/mlp-heavy',
        'research/connections/runtime','research/connections/toolchain'])
    report=HERE/'validation.json';base.write(report,result)
    base.command(['git','add',str(report)])
    base.command(['git','commit','-m','Validate MLP-heavy replacement and matched FA3 throughput'])
    head=base.command(['git','rev-parse','HEAD'])
    try:runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']()
    except Exception:
        assert base.command(['git','ls-remote','origin','refs/heads/main']).split()[0]==head
    assert base.command(['git','ls-remote','origin','refs/heads/main']).split()[0]==head
    base.write(root/'published.json',dict(commit=head,gate_commit=state['source']['commit'],job=state['job'],remote_verified=True))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('action',choices=('submit','monitor'))
    args=p.parse_args();root=args.root.resolve()
    if args.action=='submit':base.submit(root,HERE)
    else:
        base.audit_publish=audit_publish
        submit=base.submit_training
        base.submit_training=lambda root,state,summary:submit(root,state,summary,HERE)
        base.monitor(root, poll_seconds=300)
