"""Bounded native layout checks and 25-to60 restart before the single formal run."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

CHECKS = dict(tests=10, failures=0, errors=0, skipped=0)
WINDOWS = [[2048,0]]*2+[[512,0]]*6


def validate_marker(v, source, microbatch):
    assert v['complete'] and v['preflight'] and not v['formal_training']
    assert v['source']==source and v['microbatch']==microbatch and v['capability']==[9,0]
    assert v['checks']==CHECKS
    r,c = v['restart'],v['config']
    assert r['complete'] and r['steps']==60 and r['training_tokens']==60*32768
    assert r['source']==source and r['variant']=='baseline'
    assert math.isfinite(r['final_dev_ce']) and r['final_dev_ce']<r['initial_dev_ce']
    assert r['peak_reserved_bytes']<.9*22000*1024**2
    assert c['logging_backend']=='local' and c['parameters']==125829354
    assert c['gpu_capability']==[9,0] and c['source']==source and c['actual_attention_backend']=='fa3'
    assert c['model']['architecture']=='native' and c['model']['variant']=='baseline' and c['model']['streams']==1
    assert c['model']['window_pattern']=='LLSSSSSS' and c['model']['window_final_full'] is False
    assert c['protocol']['microbatch']==microbatch and c['protocol']['schedule_steps']==305176
    assert v['all_updates_finite'] and v['final_weights_finite'] and v['restored_from_step']==25
    for pattern,p in v['profiles'].items():
        assert p['complete'] and p['status']=='fits' and p['parameters']==125829354
        assert p['model']['architecture']=='native' and p['model']['window_pattern']==pattern
        assert p['source']==source and p['microbatch']==microbatch and p['actual_attention_backend']=='fa3'
        assert p['optimizer_updates']==60 and p['causality_passed']
        assert math.isfinite(p['mean_seconds']) and p['mean_seconds']>0
        assert len(p['losses'])==60 and all(math.isfinite(x) for x in p['losses'])
    assert set(v['profiles'])=={'SSSL','LLSSSSSS'}
    assert v['profiles']['LLSSSSSS']['window_sizes']==WINDOWS
    assert sorted(v['profiles']['SSSL']['window_sizes'])==sorted(WINDOWS)


def run(args, expected=0):
    result=subprocess.run([sys.executable,*args])
    assert result.returncode==expected,(args,result.returncode,expected)


def main():
    p=argparse.ArgumentParser();p.add_argument('--microbatch',type=int,required=True);a=p.parse_args()
    source=json.loads(Path('source.json').read_text())
    device=json.loads(Path('results/device.json').read_text())
    assert device['capability']==[9,0] and any(n in device['gpu'] for n in ('H100','H200'))
    assert source['architecture']=='native' and source['window_pattern']=='LLSSSSSS'
    assert source['window_final_full'] is False and source['microbatch']==a.microbatch
    marker=Path('results/native-frontload-preflight.json')
    if marker.exists():
        validate_marker(json.loads(marker.read_text()),source,a.microbatch)
        print('PASS reuse successful native LLSSSSSS preflight for this run',flush=True)
        return
    os.environ['WANDB_MODE']='disabled'
    root=Path('results/native-frontload-preflight')
    checkpoint=Path('../native-frontload-check-checkpoints')
    for path in (root,checkpoint):
        if path.exists():shutil.rmtree(path)
    root.mkdir(parents=True)
    run(['-m','pytest','-v','tests/test_native_frontload.py','--junitxml='+str(root/'tests.xml')])
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in CHECKS}
    assert counts==CHECKS,counts
    profiles={}
    for pattern in ('SSSL','LLSSSSSS'):
        out=root/(pattern+'-profile.json')
        run(['research/connections/native-frontload/profile.py','--window-pattern',pattern,
             '--sequence-length','2048','--microbatch',str(a.microbatch),
             '--attention-backend','hardware_auto','--data-dir','../data','--output',str(out)])
        profiles[pattern]=json.loads(out.read_text())
        assert profiles[pattern]['status']=='fits',profiles[pattern]
    output=root/'restart'
    args=['-m','scripts.connection_train','--architecture','native','--variant','baseline','--streams','1',
          '--window-pattern','LLSSSSSS','--no-window-final-full','--matched-backbone-init',
          '--steps','60','--schedule-steps','305176','--microbatch',str(a.microbatch),
          '--data-dir','../data','--attention-backend','hardware_auto','--output-dir',str(output),
          '--checkpoint-dir',str(checkpoint),'--run-id','native-frontload-check',
          '--segment-seconds','3600','--segment-steps','25']
    run(args,85)
    assert json.loads((output/'progress.json').read_text())['step']==25
    run(args+['--resume','auto','--segment-steps','0'])
    rows=[json.loads(line) for line in (output/'steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,61)) and all(math.isfinite(r['loss']) for r in rows)
    # No CUDA-owning parent while subprocesses execute.
    import torch
    state=torch.load(checkpoint/'latest.pt',map_location='cpu',weights_only=False)
    assert state['step']==60 and all(torch.isfinite(v).all() for v in state['model'].values())
    v=dict(complete=True,preflight=True,formal_training=False,source=source,microbatch=a.microbatch,
           capability=[9,0],checks=counts,profiles=profiles,
           restart=json.loads((output/'summary.json').read_text()),config=json.loads((output/'config.json').read_text()),
           all_updates_finite=True,final_weights_finite=True,restored_from_step=25)
    validate_marker(v,source,a.microbatch)
    del state
    shutil.rmtree(checkpoint)
    tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(marker)
    print(json.dumps(dict(preflight='passed',checks=counts,
        mean_step_ms={k:1000*p['mean_seconds'] for k,p in profiles.items()},
        overhead_percent=100*(profiles['LLSSSSSS']['mean_seconds']/profiles['SSSL']['mean_seconds']-1))),flush=True)
    print('PASS native LLSSSSSS; diagnostic weights deleted; formal run starts from seed42',flush=True)


if __name__=='__main__':main()
