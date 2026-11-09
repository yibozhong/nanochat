"""Pure-stdlib AP checks: fail-closed publication, scheduling, formal-step parser."""
import copy
import json
from pathlib import Path
import tempfile
import subprocess
from validation import validate
from preflight import validate_marker
from policy import desired_fallback,require_gpus
from watch_train import first_formal_step
from submit_train import submission_text


def main():
    # Reuse only a real report's schema as a synthetic fixture, not GPU evidence.
    root=Path(__file__).resolve().parents[3]
    text=(root/'research/connections/skip-scale/validation.json').read_text()
    v=json.loads(text.replace('skip_scale','kalman_residual').replace('58728448','58744832'))
    v['checks']['tests']=21
    v['trained_noise_statistics']={f'blocks.{i//2}.{("attn" if i%2==0 else "mlp")}_noise.{name}':
        dict(min=-.1,max=.1,mean_abs_change=.01,channel_std=.01)
        for i in range(16) for name in ('raw_q','bias','slope')}
    v['baseline_restart']=copy.deepcopy(v['restart']);v['baseline_restart']['variant']='baseline'
    validate(v,v['source'])
    mutations=[lambda w:w.update(complete=False),lambda w:w.update(formal_training=True),
        lambda w:w.update(preflight=True),lambda w:w['checks'].update(failures=1),
        lambda w:w['checks'].update(tests=19),lambda w:w.update(microbatch=32),
        lambda w:w.update(capability=[8,9]),lambda w:w['restart'].update(steps=127),
        lambda w:w['restart'].update(final_dev_ce=float('nan')),lambda w:w['restart'].update(variant='baseline'),
        lambda w:w['baseline_restart'].update(complete=False),lambda w:w['source'].update(commit='bad'),
        lambda w:w['trained_noise_statistics'].pop(next(iter(w['trained_noise_statistics']))),
        lambda w:next(iter(w['trained_noise_statistics'].values())).update(min=float('nan')),
        lambda w:[s.update(mean_abs_change=0) for s in w['trained_noise_statistics'].values()],
        lambda w:next(iter(w['profiles'].values())).update(causality_passed=False),
        lambda w:next(iter(w['profiles'].values())).update(optimizer_updates=59),
        lambda w:next(iter(w['profiles'].values())).update(gpu_uuid='different'),
        lambda w:next(iter(w['profiles'].values())).update(step_seconds=[-1]*40)]
    for mutate in mutations:
        bad=copy.deepcopy(v);mutate(bad)
        try:validate(bad,v['source'])
        except (AssertionError,KeyError):pass
        else:raise AssertionError('Invalid report accepted')
    pre=copy.deepcopy(v);pre.update(preflight=True);pre['restart']['steps']=60
    validate_marker(pre,pre['source'],pre['microbatch'],pre['capability'])
    for key,value in [('complete',False),('preflight',False),('formal_training',True),('microbatch',32)]:
        bad=copy.deepcopy(pre);bad[key]=value
        try:validate_marker(bad,pre['source'],pre['microbatch'],pre['capability'])
        except AssertionError:pass
        else:raise AssertionError('Invalid preflight accepted')
    assert desired_fallback(dict(JobStatus=1,EnteredCurrentStatus=100),3699) is False
    assert desired_fallback(dict(JobStatus=1,EnteredCurrentStatus=100),3700) is True
    assert desired_fallback(dict(JobStatus=1),10000) is False
    for status in (2,6,7):assert desired_fallback(dict(JobStatus=status),10000) is False
    for status in (3,4,5):assert desired_fallback(dict(JobStatus=status),10000) is None
    assert 'A100' not in require_gpus(False) and 'RTX PRO 6000' in require_gpus(True)
    row=json.dumps(dict(step=10,loss=6.0));marker='FORMAL_TRAINING_BEGIN kalman_residual abc'
    assert first_formal_step([row],'abc') is None
    assert first_formal_step([marker,row],'abc')['step']==10
    assert first_formal_step([marker,row],'other') is None
    assert first_formal_step([marker,json.dumps(dict(step=1,loss=float('nan')))],'abc') is None
    with tempfile.TemporaryDirectory(prefix='kalman-dryrun-') as temp:
        target=Path(temp);(target/'train.sh').write_text('#!/bin/bash\nexit 0\n');(target/'train.sh').chmod(0o700)
        (target/'logs').mkdir();(target/'results').mkdir()
        text=submission_text('synthetic',16,[])
        text='\n'.join(line for line in text.splitlines() if not line.startswith(('transfer_input_files','encrypt_input_files')))+'\n'
        (target/'train.sub').write_text(text)
        subprocess.run(['condor_submit','-dry-run','dryrun.ad','train.sub'],cwd=target,check=True,capture_output=True)
        ad=(target/'dryrun.ad').read_text()
        assert 'kalmanresidualtrainkey' in ad.lower() and 'successcheckpointexitcode=85' in ad.lower().replace(' ',''), [line for line in ad.splitlines() if 'checkpoint' in line.lower() or 'key' in line.lower()]
    print('PASS synthetic report +19 rejection cases, preflight +4 rejections,10 hardware boundaries,4 formal-step cases,Condor dry-run')


if __name__=='__main__':main()
