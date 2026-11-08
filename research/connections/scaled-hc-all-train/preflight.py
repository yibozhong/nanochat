"""One focused Hopper regression and a short restart before fresh formal training."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def validate_marker(value, source, microbatch):
    assert value['complete'] and not value['formal_training']
    assert value['source'] == source and value['microbatch'] == microbatch == 16
    assert value['capability'] == [9, 0]
    assert value['checks'] == dict(tests=1, failures=0, errors=0, skipped=0)
    result = value['restart']
    assert result['complete'] and result['source'] == source
    assert result['variant'] == 'scaled_hc_all' and result['steps'] == 60
    assert math.isfinite(result['final_dev_ce']) and result['final_dev_ce'] < result['initial_dev_ce']
    assert result['peak_reserved_bytes'] < .9 * 22000 * 1024**2
    assert value['checkpoint_finite'] and value['diagnostic_weights_deleted']


def run(args, root, name, expected=0):
    print('BEGIN hopper-preflight ' + name, flush=True)
    with (root / (name + '.log')).open('w') as log:
        proc = subprocess.Popen([sys.executable, *args], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            print(line, end='', flush=True)
        proc.wait()
    assert proc.returncode == expected, (name, proc.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--microbatch', type=int, required=True)
    args = parser.parse_args()
    source = json.loads(Path('source.json').read_text())
    device = json.loads(Path('results/device.json').read_text())
    assert source['formal_training'] and source['model_variant'] == 'scaled_hc_all'
    assert args.microbatch == source['microbatch'] == 16
    assert device['capability'] == [9, 0] and any(n in device['gpu'] for n in ('H100', 'H200'))
    marker = Path('results/scaled-hc-all-hopper-preflight.json')
    if marker.exists():
        validate_marker(json.loads(marker.read_text()), source, args.microbatch)
        print('Hopper preflight already passed for this source; resuming the same formal run', flush=True)
        return
    root = Path('results/scaled-hc-all-hopper-preflight')
    ckpt = Path('../scaled-hc-all-hopper-check-checkpoints')
    # These paths belong only to this diagnostic. Formal ../checkpoints is separate.
    for path in (root, ckpt):
        if path.exists():
            shutil.rmtree(path)
    root.mkdir(parents=True)
    os.environ['WANDB_MODE'] = 'disabled'
    run(['-m', 'pytest', '-v',
         'tests/test_scaled_hc_all.py::test_fullmodel_causal_cache_cce_recompute',
         '--junitxml=' + str(root / 'tests.xml')], root, 'recomputation')
    suites = ET.parse(root / 'tests.xml').getroot().findall('testsuite')
    counts = {k: sum(int(s.get(k, 0)) for s in suites)
              for k in ('tests', 'failures', 'errors', 'skipped')}
    assert counts == dict(tests=1, failures=0, errors=0, skipped=0)
    training = ['-m', 'scripts.connection_train', '--architecture', 'vanilla',
                '--variant', 'scaled_hc_all', '--matched-backbone-init', '--seed', '42',
                '--steps', '60', '--schedule-steps', '305176', '--sequence-length', '2048',
                '--microbatch', str(args.microbatch), '--attention-backend', 'hardware_auto',
                '--data-dir', '../data', '--output-dir', str(root / 'restart'),
                '--checkpoint-dir', str(ckpt), '--run-id', 'scaled-hc-all-hopper-restart',
                '--segment-seconds', '3600', '--segment-steps', '25']
    run(training, root, 'restart-first', 85)
    assert json.loads((root / 'restart/progress.json').read_text())['step'] == 25
    run(training + ['--resume', 'auto', '--segment-steps', '0'], root, 'restart-second')
    result = json.loads((root / 'restart/summary.json').read_text())
    config = json.loads((root / 'restart/config.json').read_text())
    rows = [json.loads(line) for line in (root / 'restart/steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows] == list(range(1, 61))
    assert all(math.isfinite(r['loss']) for r in rows)
    assert config['logging_backend'] == 'local' and config['actual_attention_backend'] == 'fa3'
    assert config['parameters'] == 59615669 and config['protocol']['schedule_steps'] == 305176
    # Delay torch import until all CUDA children exit, including exclusive-process GPUs.
    import torch
    state = torch.load(ckpt / 'latest.pt', map_location='cpu', weights_only=False)
    assert all(torch.isfinite(t).all() for t in state['model'].values())
    scales = [t for n, t in state['model'].items()
              if n.endswith(('.input_scale', '.read_scale', '.post_scale'))]
    assert len(scales) == 49 and max((t - 1).abs().max().item() for t in scales) > 1e-6
    shutil.rmtree(ckpt)
    value = dict(complete=True, formal_training=False, source=source, microbatch=args.microbatch,
                 capability=device['capability'], gpu=device['gpu'], checks=counts, restart=result,
                 checkpoint_finite=True, diagnostic_weights_deleted=True)
    validate_marker(value, source, args.microbatch)
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(marker)
    print('PASS Hopper preflight; diagnostic weights deleted; ready for fresh formal initialization', flush=True)


if __name__ == '__main__':
    main()
