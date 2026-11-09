"""Kalman residual GPU correctness, ABBA throughput and two real NTP restarts."""
import argparse
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET


def run(args, root, name, expected=0):
    print('BEGIN '+name,flush=True)
    with (root/(name+'.log')).open('w') as log:
        proc=subprocess.Popen([sys.executable,*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:
            log.write(line);print(line,end='',flush=True)
        proc.wait()
    assert proc.returncode==expected,(name,proc.returncode)
    print('PASS '+name,flush=True)


def checks(root):
    run(['-m','pytest','-v','tests/test_kalman_residual.py','--junitxml='+str(root/'tests.xml')],root,'tests')
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=20,failures=0,errors=0,skipped=0),counts
    return counts


def restart(root,variant,micro,steps):
    checkpoint_dir=Path('../kalman-residual-check-checkpoints')/variant
    output=root/('restart' if variant=='kalman_residual' else 'baseline-restart')
    shutil.rmtree(output,ignore_errors=True);shutil.rmtree(checkpoint_dir,ignore_errors=True)
    args=['-m','scripts.connection_train','--variant',variant,'--streams','1','--matched-backbone-init',
          '--steps',str(steps),'--schedule-steps','305176','--microbatch',str(micro),
          '--data-dir','../data','--attention-backend','hardware_auto',
          '--output-dir',str(output),'--checkpoint-dir',str(checkpoint_dir),
          '--run-id',variant+'-restart-check','--segment-seconds','3600','--segment-steps','25']
    run(args,root,variant+'-restart-first',85)
    assert json.loads((output/'progress.json').read_text())['step']==25
    run(args+['--resume','auto','--segment-steps','0'],root,variant+'-restart-second')
    summary=json.loads((output/'summary.json').read_text());config=json.loads((output/'config.json').read_text())
    rows=[json.loads(s) for s in (output/'steps.jsonl').read_text().splitlines()]
    assert summary['complete'] and summary['steps']==steps
    assert [r['step'] for r in rows]==list(range(1,steps+1)) and all(math.isfinite(r['loss']) for r in rows)
    assert summary['final_dev_ce']<summary['initial_dev_ce'] and math.isfinite(summary['final_dev_ce'])
    assert config['logging_backend']=='local' and config['protocol']['schedule_steps']==305176
    assert config['model']['streams']==1 and config['parameters']==(58744832 if variant=='kalman_residual' else 58720256)
    assert summary['peak_reserved_bytes']<.9*22000*1024**2
    return summary,config,checkpoint_dir


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--microbatch',type=int,default=16)
    a=p.parse_args();root=Path('results/kalman-residual-preflight' if a.preflight else 'results/kalman-residual')
    root.mkdir(parents=True,exist_ok=True);counts=checks(root);profiles={};micro=a.microbatch
    if not a.preflight:
        for micro in (16,8,4):
            profiles={}
            for index,variant in enumerate(('baseline','kalman_residual','kalman_residual','baseline')):
                name=f'profile-{index}-{variant}-{micro}'
                run(['-m','scripts.connection_context_profile','--variant',variant,'--sequence-length','2048',
                     '--microbatch',str(micro),'--attention-backend','hardware_auto','--output',str(root/(name+'.json'))],root,name)
                profiles[name]=json.loads((root/(name+'.json')).read_text())
            if all(v['complete'] and v['status']=='fits' for v in profiles.values()):break
            assert all(v['status'] in ('fits','oom','over_budget') for v in profiles.values()),profiles
        else:raise RuntimeError('No common microbatch fits22GB')
        assert len({v['gpu_uuid'] for v in profiles.values()})==1
        assert len({v['input_sha256'] for v in profiles.values()})==1
        assert all(v['optimizer_updates']==60 and v['causality_passed'] for v in profiles.values())
    steps=60 if a.preflight else 128
    baseline=None
    if not a.preflight:baseline,_,_=restart(root,'baseline',micro,steps)
    summary,config,checkpoint_dir=restart(root,'kalman_residual',micro,steps)
    # Import Torch only after all GPU subprocesses have exited (exclusive GPUs).
    import torch
    from nanochat.kalman_residual import DiagonalNoise
    state=torch.load(checkpoint_dir/'latest.pt',map_location='cpu',weights_only=False)
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    init=DiagonalNoise(512).state_dict()
    noise={k:dict(min=v.min().item(),max=v.max().item(),
                 mean_abs_change=(v-init[k.rsplit('.',1)[1]]).abs().mean().item(),channel_std=v.std().item())
           for k,v in state['model'].items() if '_noise.' in k}
    assert len(noise)==48
    assert sum(k.endswith('.slope') and s['mean_abs_change']>1e-7 and s['channel_std']>1e-7 for k,s in noise.items())==16
    result=dict(complete=True,formal_training=False,preflight=a.preflight,checks=counts,
        source=json.loads(Path('source.json').read_text()),microbatch=micro,gpu=config['gpu'],capability=config['gpu_capability'],
        profiles=profiles,restart=summary,baseline_restart=baseline,trained_noise_statistics=noise,
        limitations='Short finite training and restart do not establish10B stability or quality; learned variance is not calibrated uncertainty.')
    if profiles:
        means={variant:statistics.mean(v['mean_seconds'] for v in profiles.values() if v['variant']==variant)
               for variant in ('baseline','kalman_residual')}
        result.update(mean_step_seconds=means,overhead_percent=100*(means['kalman_residual']/means['baseline']-1),
                      pilot_dev_difference=summary['final_dev_ce']-baseline['final_dev_ce'])
    shutil.rmtree('../kalman-residual-check-checkpoints')
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','restart','baseline_restart','source')}),flush=True)


if __name__=='__main__':main()
