"""Small AP-safe checks for scheduling boundaries and automatic-launch rejection."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from policy import desired_fallback, require_gpus
from watch_train import first_formal_step
from submit_train import submission_text, execution_files
from validation import validate
from preflight import validate_marker


def rejected(fn,*args):
    try:fn(*args)
    except (AssertionError,KeyError,ValueError):return
    raise AssertionError('Invalid result was accepted')


def main():
    repo=Path(__file__).resolve().parents[3]
    for status,age,want in [(1,0,False),(1,3599,False),(1,3600,True),(1,7200,True),(2,7200,False),(5,7200,None),(6,7200,False),(1,-1,False)]:
        assert desired_fallback(dict(JobStatus=status,EnteredCurrentStatus=10000-age),10000)==want
    assert desired_fallback(dict(JobStatus=1),10000)==False
    lines=['{"step":25,"loss":6.5}','FORMAL_TRAINING_BEGIN attnres_scale other','{"step":1,"loss":8.0}']
    assert first_formal_step(lines,'run') is None
    assert first_formal_step(lines+['FORMAL_TRAINING_BEGIN attnres_scale run','{"step":1,"loss":NaN}'],'run') is None
    assert first_formal_step(lines+['FORMAL_TRAINING_BEGIN attnres_scale run','{"step":1,"loss":10.3}'],'run')['step']==1
    # Independent sentinel result checks the publication guard against common
    # failure modes; it is never persisted as a real GPU result.
    source={'commit':'synthetic'}
    restart=dict(complete=True,steps=128,variant='attnres_scale',source=source,training_tokens=128*32768,initial_dev_ce=10.,final_dev_ce=6.,peak_reserved_bytes=1)
    v=dict(complete=True,formal_training=False,preflight=False,source=source,capability=[9,0],microbatch=16,
        checks=dict(tests=24,failures=0,errors=0,skipped=0),restart=restart,
        trained_scale_statistics={str(i):dict(min=.9,max=1.1,mean_abs_change=.01,channel_std=.01) for i in range(16)},profiles={})
    for i,var in enumerate(('baseline','attnres','attnres_scale','attnres_scale','attnres','baseline')):
        v['profiles'][str(i)]=dict(variant=var,gpu_uuid='same',input_sha256='same',complete=True,status='fits',source=source,
            optimizer_updates=60,warmup_steps=20,timed_steps=40,causality_passed=True,microbatch=16,batch_tokens=32768,sequence_length=2048,
            logging_backend='local',model=dict(architecture='vanilla',streams=1),parameters=({'attnres_scale':58782720,'attnres':58737664,'baseline':58720256}[var]),
            losses=[6.]*60,step_seconds=[.1]*40,mean_seconds=.1,peak_reserved_bytes=1)
    validate(v,source)
    changes=[lambda x:x.update(complete=False),lambda x:x['checks'].update(failures=1),
        lambda x:x['checks'].update(skipped=1),lambda x:x['restart'].update(final_dev_ce=float('nan')),
        lambda x:x['restart'].update(steps=60),lambda x:x['profiles']['2']['losses'].__setitem__(3,float('inf')),
        lambda x:x['profiles']['3'].update(gpu_uuid='different'),lambda x:x['profiles']['2']['model'].update(streams=4),
        lambda x:x['profiles']['2'].update(parameters=59615669),lambda x:x['trained_scale_statistics']['0'].update(channel_std=0.)]
    for change in changes:
        bad=copy.deepcopy(v);change(bad);rejected(validate,bad,source)
    marker=copy.deepcopy(v);marker.update(preflight=True);marker['restart']['steps']=60
    validate_marker(marker,source,16,[9,0]);rejected(validate_marker,marker,source,16,[12,0])
    rejected(validate_marker,marker,{'commit':'other'},16,[9,0])
    state_path=Path('/home/yzhong239/attnres-scale-gate-20261108-r2/state.json')
    gate=json.loads(state_path.read_text()) if state_path.exists() else None
    files=execution_files(gate['source']['commit']) if gate else None
    for p in [*Path(__file__).parent.glob('*.py'),repo/'tests/test_attnres_scale.py',repo/'scripts/connection_attnres_scale_gate.py']:
        ast.parse(p.read_text(),filename=str(p))
    for p in Path(__file__).parent.glob('*.sh'):
        subprocess.run(['bash','-n',str(p)],check=True)
        inside=None;lines=[]
        for line in p.read_text().splitlines():
            if inside:
                if line==inside:ast.parse('\n'.join(lines),filename=str(p));inside=None
                else:lines.append(line)
            elif "<<'" in line:inside=line.split("<<'",1)[1].split("'",1)[0];lines=[]
    with tempfile.TemporaryDirectory(prefix='attnres-scale-dryrun-') as directory:
        d=Path(directory);(d/'logs').mkdir();(d/'results').mkdir()
        (d/'train.sh').write_text((Path(__file__).parent/'train.sh').read_text());(d/'train.sh').chmod(0o755)
        text=submission_text('STATIC00',16,['/home/yzhong239/.netrc'])
        for name,fallback in [('initial',False),('fallback',True)]:
            value='\n'.join(('require_gpus = '+require_gpus(fallback)) if l.startswith('require_gpus =') else l for l in text.splitlines())+'\n'
            p=d/(name+'.sub');p.write_text(value)
            subprocess.run(['condor_submit','-dry-run',str(d/(name+'.ad')),str(p)],cwd=d,text=True,capture_output=True,check=True)
            ad=(d/(name+'.ad')).read_text().replace(' ','')
            assert 'SuccessCheckpointExitCode=85' in ad and 'AttnResScaleTrainKey="attnres-scale-10b-20261108"' in ad
            assert 'GPUsMinCapability=8.0' in ad and 'JobStatus=5' in ad
    value=dict(complete=True,gate_source=gate['source']['commit'] if gate else 'pending immutable submission',execution_files_unchanged=len(files) if files else None,
        queue_boundary_cases=9,formal_start_cases=3,bad_gpu_result_rejections=10,bad_family_marker_rejections=2,
        condor_dry_runs=2,gpu_tests='pending; this report is static validation only')
    (Path(__file__).parent/'static-validation.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
