"""Validate and publish post-step W_o NS, then run one controlled 10B experiment."""
import argparse,importlib.util,json,math,runpy,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('moment_workflow',HERE.parent/'moments/workflow.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.KEY='attention-output-ns-10b-20261104'
base.MODES=('output_ns',)


def audit_publish(root,state):
    with tarfile.open(root/'results'/f'gate-{state["job"]}.tar') as tf:
        result=json.load(tf.extractfile('results/gate.json'))
        assert result['complete'] and result['source']==state['source']
        assert result['checks']==dict(tests=17,failures=0,errors=0,skipped=0)
        assert not result['formal_training'] and not result['weights_returned']
        assert len(result['profiles'])==4 and len({p['gpu'] for p in result['profiles']})==1
        for p in result['profiles']:
            assert p['mode'] in ('baseline','output_ns') and p['parameters']==58720256
            assert len(p['samples_seconds'])==30 and all(math.isfinite(x) and x>0 for x in p['samples_seconds'])
            assert p['peak_reserved_bytes']<.9*22000*1024**2 and math.isfinite(p['dev_ce'])
            assert len(p['diagnostics']['layers'])==8
            if p['mode']=='output_ns':
                assert all(row['weight']['scaled_orthogonality_error']<.2 for row in p['diagnostics']['layers'])
        r=result['resume'];assert r['exact_steps_1_to_50'] and r['projection_after_resume_verified']
        assert r['summary']['complete'] and r['summary']['steps']==50
        assert r['config']['model']['attention_output_ns'] and r['config']['model']['attention_statistic']=='none'
        assert r['config']['model']['attention_kernel']=='fla_triton'
        assert r['config']['protocol']['schedule_steps']==305176 and r['config']['logging_backend']=='local'
        for member in tf.getmembers():
            assert not member.name.endswith(('.pt','.wandb'))
            if not member.isfile():continue
            path=Path(member.name);assert not path.is_absolute() and path.parts[0]=='results' and '..' not in path.parts
            dest=root/'collected'/Path(*path.parts[1:]);dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(tf.extractfile(member).read())
    summary={}
    for mode in ('baseline','output_ns'):
        rows=[p for p in result['profiles'] if p['mode']==mode];assert len(rows)==2
        summary[mode]=dict(mean_seconds=sum(p['mean_seconds'] for p in rows)/2,
            peak_allocated_bytes=max(p['peak_allocated_bytes'] for p in rows),
            peak_reserved_bytes=max(p['peak_reserved_bytes'] for p in rows),parameters=rows[0]['parameters'])
    for row in summary.values():
        row['time_relative_to_baseline']=row['mean_seconds']/summary['baseline']['mean_seconds']
        for key in ('allocated','reserved'):
            row[f'additional_peak_{key}_bytes']=row[f'peak_{key}_bytes']-summary['baseline'][f'peak_{key}_bytes']
    result.update(job=state['job'],summary=summary,quality_reference=dict(job='7802254.0',run_id='46bd7320',
        steps=305176,dev_ce=3.1815003156661987,test_ce=3.160942070186138,
        note='Existing matched10B Triton baseline. Gate50-step CE and spectra are functional checks, not a quality conclusion.'))
    base.write(root/'validation.json',result)
    assert not base.command(['git','status','--porcelain'])
    assert not base.command(['git','diff','--name-only',state['source']['commit'],'HEAD','--',
        'nanochat','scripts','tests','research/connections/attention-output-ns',
        'research/connections/runtime','research/connections/toolchain'])
    report=HERE/'validation.json';base.write(report,result)
    base.command(['git','add',str(report)])
    base.command(['git','commit','-m','Validate attention output weight NS projection and controlled training recovery'])
    head=base.command(['git','rev-parse','HEAD'])
    try:runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']()
    except Exception:
        assert base.command(['git','ls-remote','origin','refs/heads/main']).split()[0]==head
    assert base.command(['git','ls-remote','origin','refs/heads/main']).split()[0]==head
    base.write(root/'published.json',dict(commit=head,gate_commit=state['source']['commit'],job=state['job'],remote_verified=True))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('action',choices=('submit','monitor'))
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='submit':base.submit(root,HERE)
    else:
        base.audit_publish=audit_publish
        submit=base.submit_training
        base.submit_training=lambda root,state,summary:submit(root,state,summary,HERE)
        base.monitor(root)
