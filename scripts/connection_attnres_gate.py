"""Bounded NVIDIA AttnRes GPU checks and real NTP; never logs to W&B."""
import argparse
import json
import math
from pathlib import Path
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
    run(['-m','pytest','-v','tests/test_attnres.py',
         'tests/test_connection_hardware_backend.py',
         'tests/test_connection_training_resume.py',
         '--junitxml='+str(root/'tests.xml')],root,'tests')
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=21,failures=0,errors=0,skipped=0),counts
    return counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true')
    p.add_argument('--microbatch',type=int,default=16)
    a=p.parse_args()
    import torch
    root=Path('results/attnres' if not a.preflight else 'results/attnres-preflight')
    root.mkdir(parents=True,exist_ok=True)
    counts=checks(root)
    profiles={};micro=a.microbatch
    if not a.preflight:
        for micro in (16,8,4):
            profiles={}
            for variant in ('baseline','attnres'):
                name=f'profile-{variant}-{micro}'
                run(['-m','scripts.connection_context_profile','--variant',variant,
                     '--sequence-length','2048','--microbatch',str(micro),'--attention-backend','hardware_auto',
                     '--output',str(root/(name+'.json'))],root,name)
                profiles[variant]=json.loads((root/(name+'.json')).read_text())
            if all(v['complete'] and v['status']=='fits' for v in profiles.values()):break
            assert all(v['status'] in ('fits','oom','over_budget') for v in profiles.values()),profiles
        else:raise RuntimeError('No common microbatch fits22GB')
        assert len({v['gpu_uuid'] for v in profiles.values()})==1
        assert len({v['input_sha256'] for v in profiles.values()})==1
        assert profiles['attnres']['parameters']==58737664
        assert all(v['optimizer_updates']==60 and v['causality_passed'] for v in profiles.values())
    # Real NTP, independent processes, exact scheduler checkpoint boundary.
    steps=60 if a.preflight else 128
    args=['-m','scripts.connection_train','--variant','attnres','--matched-backbone-init',
          '--steps',str(steps),'--schedule-steps','305176','--microbatch',str(micro),
          '--data-dir','../data','--attention-backend','hardware_auto',
          '--output-dir',str(root/'restart'),'--checkpoint-dir','../attnres-check-checkpoints',
          '--run-id','attnres-restart-check','--segment-seconds','3600','--segment-steps','25']
    run(args,root,'restart-first',85)
    assert json.loads((root/'restart/progress.json').read_text())['step']==25
    run(args+['--resume','auto','--segment-steps','0'],root,'restart-second')
    summary=json.loads((root/'restart/summary.json').read_text())
    config=json.loads((root/'restart/config.json').read_text())
    rows=[json.loads(s) for s in (root/'restart/steps.jsonl').read_text().splitlines()]
    assert summary['complete'] and summary['steps']==steps
    assert [r['step'] for r in rows]==list(range(1,steps+1)) and all(math.isfinite(r['loss']) for r in rows)
    assert summary['final_dev_ce']<summary['initial_dev_ce'] and math.isfinite(summary['final_dev_ce'])
    assert config['logging_backend']=='local' and config['protocol']['schedule_steps']==305176
    state=torch.load('../attnres-check-checkpoints/latest.pt',map_location='cpu',weights_only=False)
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    assert summary['peak_reserved_bytes']<.9*22000*1024**2
    result=dict(complete=True,formal_training=False,preflight=a.preflight,checks=counts,
        source=json.loads(Path('source.json').read_text()),microbatch=micro,
        gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
        profiles=profiles,restart=summary,
        limitations='Short finite training and restart do not establish10B stability or quality.')
    if profiles:
        b,n=profiles['baseline'],profiles['attnres']
        result.update(overhead_percent=100*(n['mean_seconds']/b['mean_seconds']-1),
            additional_peak_allocated_bytes=n['peak_allocated_bytes']-b['peak_allocated_bytes'],
            additional_peak_reserved_bytes=n['peak_reserved_bytes']-b['peak_reserved_bytes'])
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','restart','source')}),flush=True)


if __name__=='__main__':main()
