"""Reuse the validated Hopper family; validate a newly encountered GPU family."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def validate_marker(v, source, microbatch, capability):
    assert v['complete'] and v['preflight'] and not v['formal_training']
    assert v['source'] == source and v['microbatch'] == microbatch and v['capability'] == capability
    assert v['checks'] == dict(tests=30, failures=0, errors=0, skipped=0)
    r, c = v['restart'], v['config']
    assert r['complete'] and r['steps'] == 60 and r['training_tokens'] == 60*32768
    assert r['source'] == source and r['variant'] == 'baseline'
    assert math.isfinite(r['final_dev_ce']) and r['final_dev_ce'] < r['initial_dev_ce']
    assert r['peak_reserved_bytes'] < .9*22000*1024**2
    assert c['logging_backend'] == 'local' and c['parameters'] == 58720256
    assert c['gpu_capability'] == capability and c['source'] == source
    assert c['model']['attention_pattern'] == 'progressive_dilated' and c['model']['sparse_backend'] == 'triton'
    assert c['model']['streams'] == 1 and c['model']['architecture'] == 'vanilla'
    assert (c['model']['sparse_budget'],c['model']['sparse_core'],c['model']['sparse_sinks']) == (256,128,4)
    assert c['protocol']['microbatch'] == microbatch and c['protocol']['schedule_steps'] == 305176
    assert v['all_updates_finite'] and v['final_weights_finite'] and v['restored_from_step'] == 25


def run(args, expected=0):
    result = subprocess.run([sys.executable, *args])
    assert result.returncode == expected, (args, result.returncode, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--microbatch', type=int, required=True)
    a = parser.parse_args()
    source = json.loads(Path('source.json').read_text())
    device = json.loads(Path('results/device.json').read_text())
    cap, gpu = device['capability'], device['gpu']
    assert (cap == [9,0] and any(n in gpu for n in ('H100','H200'))) or (cap == [8,0] and 'A100' in gpu) or (cap == [12,0] and gpu == 'NVIDIA RTX PRO 6000 Blackwell Server Edition')
    assert source['formal_training'] and source['model_variant'] == 'baseline'
    assert source['attention_pattern'] == 'progressive_dilated' and source['microbatch'] == a.microbatch
    if cap in source['passed_families']:
        print('Reusing the successful 30-check dilated attention gate and 25-to128 restore for this family', flush=True)
        return
    marker = Path('results') / ('dilated-family-' + '.'.join(map(str,cap)) + '.json')
    if marker.exists():
        validate_marker(json.loads(marker.read_text()), source, a.microbatch, cap)
        print('Reusing successful new-family preflight for this run', flush=True)
        return
    os.environ['WANDB_MODE'] = 'disabled'
    root = Path('results/dilated-family-preflight')
    # An interrupted preflight can be rerun locally; no diagnostic weights
    # ever enter the separate formal checkpoints directory.
    checkpoint = Path('../dilated-family-check-checkpoints')
    for path in (root, checkpoint):
        if path.exists():
            shutil.rmtree(path)
    root.mkdir(parents=True)
    run(['-m','pytest','-v','tests/test_dilated_attention.py','--junitxml='+str(root/'tests.xml')])
    suites = ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts = {k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts == dict(tests=30,failures=0,errors=0,skipped=0)
    output = root/'restart'
    args = ['-m','scripts.connection_train','--variant','baseline','--streams','1',
            '--attention-pattern','progressive_dilated','--sparse-budget','256','--sparse-core','128',
            '--sparse-sinks','4','--sparse-backend','triton','--matched-backbone-init',
            '--steps','60','--schedule-steps','305176','--microbatch',str(a.microbatch),
            '--data-dir','../data','--attention-backend','hardware_auto','--output-dir',str(output),
            '--checkpoint-dir',str(checkpoint),'--run-id','dilated-family-check',
            '--segment-seconds','3600','--segment-steps','25']
    run(args,85)
    assert json.loads((output/'progress.json').read_text())['step'] == 25
    run(args+['--resume','auto','--segment-steps','0'])
    rows = [json.loads(line) for line in (output/'steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows] == list(range(1,61)) and all(math.isfinite(r['loss']) for r in rows)
    # Import Torch only after child CUDA processes have exited.
    import torch
    state = torch.load(checkpoint/'latest.pt', map_location='cpu', weights_only=False)
    assert state['step'] == 60 and all(torch.isfinite(v).all() for v in state['model'].values())
    v = dict(complete=True, preflight=True, formal_training=False, source=source,
             microbatch=a.microbatch, capability=cap, checks=counts,
             restart=json.loads((output/'summary.json').read_text()),
             config=json.loads((output/'config.json').read_text()),
             all_updates_finite=True, final_weights_finite=True, restored_from_step=25)
    validate_marker(v,source,a.microbatch,cap)
    del state
    shutil.rmtree(checkpoint)
    tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(marker)
    print('PASS new-family preflight; diagnostic weights deleted',flush=True)


if __name__ == '__main__':
    main()
