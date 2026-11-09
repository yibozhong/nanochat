"""Reuse the passed Hopper gate after checking frozen source and model identity."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(source,device,microbatch):
    assert device['capability']==[9,0] and any(n in device['gpu'] for n in ('H100','H200'))
    assert source['formal_training'] and source['model_variant']=='baseline'
    assert source['experiment']=='sssl_swa_loop3' and source['microbatch']==microbatch
    assert source['passed_families']==[[9,0]]
    assert source['model_options']==dict(architecture='vanilla',variant='baseline',streams=1,
        window_pattern='SSSL',swa_looping=True,swa_loops=3,swa_window=512)
    assert source['initialization']=='seed42_from_zero_no_validation_weights'
    assert source['steps']==305176 and source['training_tokens']==10000007168


def main():
    p=argparse.ArgumentParser();p.add_argument('--microbatch',type=int,required=True);a=p.parse_args()
    source=json.loads(Path('source.json').read_text())
    device=json.loads(Path('results/device.json').read_text())
    validate(source,device,a.microbatch)
    for name,digest in source['validated_execution_files'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    print('Reusing21passed Hopper checks and25-to128 NTP restart; execution hashes match. No diagnostic weights loaded.',flush=True)


if __name__=='__main__':main()
