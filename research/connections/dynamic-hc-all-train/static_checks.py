"""AP-safe submission checks; reuse the published GPU results without rerunning them."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import tempfile
from policy import desired_fallback, require_gpus
from watch_train import first_formal_step
from submit_train import submission_text, execution_files, GATE_ROOT
from validation import validate
from preflight import validate_marker


def rejected(fn,*args):
    try:fn(*args)
    except (AssertionError,KeyError,ValueError):return
    raise AssertionError('Invalid result was accepted')


def main():
    here=Path(__file__).parent
    repo=here.parents[2]
    for status,age,want in [(1,0,False),(1,3599,False),(1,3600,True),(1,7200,True),(2,7200,False),(5,7200,None),(6,7200,False),(1,-1,False)]:
        assert desired_fallback(dict(JobStatus=status,EnteredCurrentStatus=10000-age),10000)==want
    assert desired_fallback(dict(JobStatus=1),10000)==False
    lines=['{"step":25,"loss":6.5}','FORMAL_TRAINING_BEGIN dynamic_hc_all other','{"step":1,"loss":8.0}']
    assert first_formal_step(lines,'run') is None
    assert first_formal_step(lines+['FORMAL_TRAINING_BEGIN dynamic_hc_all run','{"step":1,"loss":NaN}'],'run') is None
    assert first_formal_step(lines+['FORMAL_TRAINING_BEGIN dynamic_hc_all run','{"step":1,"loss":10.3}'],'run')['step']==1
    v=json.loads((repo/'research/connections/dynamic-hc-all/validation.json').read_text())
    source=v['source'];validate(v,source)
    dynamic=next(k for k,p in v['profiles'].items() if p['variant']=='dynamic_hc_all')
    other=next(k for k,p in v['profiles'].items() if p['variant']=='scaled_hc_all')
    slope=next(k for k in v['trained_gate_statistics'] if k.endswith('_a'))
    changes=[lambda x:x.update(complete=False),lambda x:x['checks'].update(failures=1),
        lambda x:x['checks'].update(skipped=1),lambda x:x['restart'].update(final_dev_ce=float('nan')),
        lambda x:x['restart'].update(steps=60),lambda x:x['profiles'][dynamic]['losses'].__setitem__(3,float('inf')),
        lambda x:x['profiles'][other].update(gpu_uuid='different'),lambda x:x['profiles'][dynamic]['model'].update(streams=1),
        lambda x:x['profiles'][dynamic].update(parameters=59615669),lambda x:x['trained_gate_statistics'][slope].update(channel_std=0.)]
    for change in changes:
        bad=copy.deepcopy(v);change(bad);rejected(validate,bad,source)
    # Synthetic marker exercises the validator only; never saved as a GPU result.
    marker=copy.deepcopy(v);marker.update(preflight=True);marker['restart']['steps']=60
    validate_marker(marker,source,16,[9,0]);rejected(validate_marker,marker,source,16,[8,0])
    rejected(validate_marker,marker,{'commit':'other'},16,[9,0])
    gate=json.loads((GATE_ROOT/'state.json').read_text())
    files=execution_files(gate['source']['commit'])
    for p in here.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    for p in here.glob('*.sh'):
        subprocess.run(['bash','-n',str(p)],check=True)
        inside=None;lines=[]
        for line in p.read_text().splitlines():
            if inside:
                if line==inside:ast.parse('\n'.join(lines),filename=str(p));inside=None
                else:lines.append(line)
            elif "<<'" in line:inside=line.split("<<'",1)[1].split("'",1)[0];lines=[]
    with tempfile.TemporaryDirectory(prefix='dynamic-hc-all-dryrun-') as directory:
        d=Path(directory);(d/'logs').mkdir();(d/'results').mkdir()
        (d/'train.sh').write_text((here/'train.sh').read_text());(d/'train.sh').chmod(0o755)
        text=submission_text('STATIC00',16,['/home/yzhong239/.netrc'])
        for name,fallback in [('initial',False),('fallback',True)]:
            value='\n'.join(('require_gpus = '+require_gpus(fallback)) if l.startswith('require_gpus =') else l for l in text.splitlines())+'\n'
            p=d/(name+'.sub');p.write_text(value)
            subprocess.run(['condor_submit','-dry-run',str(d/(name+'.ad')),str(p)],cwd=d,text=True,capture_output=True,check=True)
            ad=(d/(name+'.ad')).read_text().replace(' ','')
            assert 'SuccessCheckpointExitCode=85' in ad and 'DynamicHCAllTrainKey="dynamic-hc-all-10b-20261108"' in ad
            assert 'GPUsMinCapability=8.0' in ad and 'JobStatus=5' in ad
            requirement=next(l for l in ad.splitlines() if l.startswith('RequireGPUs='))
            assert 'H100|H200' in requirement and ('A100' in requirement)==fallback and 'PRO' not in requirement
    value=dict(complete=True,gate_source=source['commit'],execution_files_unchanged=len(files),
        queue_boundary_cases=9,formal_start_cases=3,bad_gpu_result_rejections=10,bad_family_marker_rejections=2,
        condor_dry_runs=2,gpu_tests='Reuse29passed checks, six60step profiles,25-to128 restart from8087159.0; no new GPU test')
    (here/'static-validation.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
