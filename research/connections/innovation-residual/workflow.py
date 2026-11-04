"""Publish a validated optional residual experiment, then one authorized 10B run."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import runpy
import tarfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('moment_workflow',HERE.parent/'moments/workflow.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.KEY='innovation-residual-10b-20261104'
base.MODES=('innovation',)


def audit_publish(root,state):
    with tarfile.open(root/'results'/f'gate-{state["job"]}.tar') as tf:
        result=json.load(tf.extractfile('results/gate.json'))
        assert result['complete'] and result['source']==state['source']
        assert result['checks']==dict(tests=19,failures=0,errors=0,skipped=0)
        assert not result['formal_training'] and not result['weights_returned']
        assert len(result['profiles'])==4 and len({p['gpu'] for p in result['profiles']})==1
        for p in result['profiles']:
            assert p['mode'] in ('baseline','innovation')
            assert p['parameters']==(58720256 if p['mode']=='baseline' else 62922752)
            assert len(p['samples_seconds'])==30 and all(math.isfinite(x) and x>0 for x in p['samples_seconds'])
            assert p['peak_reserved_bytes']<.9*22000*1024**2 and math.isfinite(p['dev_ce'])
            if p['mode']=='innovation':
                assert p['innovation']['beta']==1. and len(p['innovation']['branches'])==16
        r=result['resume']
        assert r['exact_steps_1_to_50'] and r['beta_schedule_restored']
        assert r['summary']['complete'] and r['summary']['steps']==50
        assert r['config']['model']['innovation_residual'] and r['config']['model']['attention_statistic']=='none'
        assert r['config']['model']['attention_kernel']=='fla_triton'
        assert r['config']['protocol']['schedule_steps']==305176 and r['config']['logging_backend']=='local'
        for member in tf.getmembers():
            assert not member.name.endswith(('.pt','.wandb'))
            if not member.isfile():continue
            path=Path(member.name);assert not path.is_absolute() and path.parts[0]=='results' and '..' not in path.parts
            dest=root/'collected'/Path(*path.parts[1:]);dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(tf.extractfile(member).read())
    summary={}
    for mode in ('baseline','innovation'):
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
        note='Existing matched 10B Triton baseline. Gate 50-step CE is not a quality conclusion.'))
    base.write(root/'validation.json',result)
    assert not base.command(['git','status','--porcelain'])
    assert not base.command(['git','diff','--name-only',state['source']['commit'],'HEAD','--',
        'nanochat','scripts','tests','research/connections/innovation-residual',
        'research/connections/runtime','research/connections/toolchain'])
    report=HERE/'validation.json';base.write(report,result)
    base.command(['git','add',str(report)])
    base.command(['git','commit','-m','Validate Innovation Residual gradients, throughput and checkpoint recovery'])
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
