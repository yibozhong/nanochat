"""Reuse the initial family gate; validate any newly encountered GPU family."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def validate_marker(v,source,microbatch,capability):
    assert v['complete'] and not v['formal_training'] and v['preflight']
    assert v['source']==source and v['microbatch']==microbatch and v['capability']==capability
    assert v['checks']==dict(tests=21,failures=0,errors=0,skipped=0)
    r=v['restart']
    assert r['complete'] and r['steps']==60 and r['source']==source and r['variant']=='kalman_residual'
    assert math.isfinite(r['final_dev_ce']) and r['final_dev_ce']<r['initial_dev_ce']
    assert r['peak_reserved_bytes']<.9*22000*1024**2
    scales=v['trained_noise_statistics']
    assert len(scales)==48 and all(math.isfinite(s['min']) and math.isfinite(s['max'])  for s in scales.values())

    assert sum(k.endswith('.slope') and s['mean_abs_change']>1e-7 and s['channel_std']>1e-7 for k,s in scales.items())==16

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--microbatch',type=int,required=True);a=parser.parse_args()
    source=json.loads(Path('source.json').read_text());device=json.loads(Path('results/device.json').read_text())
    cap=device['capability'];gpu=device['gpu']
    assert (cap==[9,0] and any(n in gpu for n in ('H100','H200'))) or (cap==[8,0] and 'A100' in gpu) or (cap==[12,0] and gpu=='NVIDIA RTX PRO 6000 Blackwell Server Edition')
    assert source['formal_training'] and source['model_variant']=='kalman_residual' and source['microbatch']==a.microbatch
    for name,digest in source['validated_execution_files'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    if cap in source['passed_families']:
        print('Reusing full kalman-residual GPU gate for this family; source hashes match',flush=True)
        return
    marker=Path('results')/('kalman-residual-family-'+'.'.join(map(str,cap))+'.json')
    if marker.exists():
        validate_marker(json.loads(marker.read_text()),source,a.microbatch,cap)
        print('Reusing successful family preflight for this run',flush=True);return
    os.environ['WANDB_MODE']='disabled'
    subprocess.run([sys.executable,'-m','scripts.connection_kalman_residual_gate','--preflight','--microbatch',str(a.microbatch)],check=True)
    v=json.loads(Path('results/kalman-residual-preflight/validation.json').read_text())
    validate_marker(v,source,a.microbatch,cap)
    assert not Path('../kalman-residual-check-checkpoints').exists()
    tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(marker)
    print('PASS new-family preflight; diagnostic weights deleted',flush=True)


if __name__=='__main__':main()
