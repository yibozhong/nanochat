"""Global-softmax checks, paired cost, and real restart before the10B partition experiment."""
import json
import math
import os
from pathlib import Path
import sys
import uuid
import xml.etree.ElementTree as ET

from scripts.connection_pretrain_gate import execute


def main():
    root = Path('results')
    source = json.loads(Path('source.json').read_text())
    env = os.environ.copy()
    env.pop('NVIDIA_TF32_OVERRIDE', None)
    tests = ['tests/test_partitioned_readout.py', 'tests/test_connection_training_resume.py', 'tests/test_partition_stream_mixing.py']
    execute([sys.executable, '-m', 'pytest', *tests, '-v',
        '--junitxml=results/partition-mix-tests.xml'], root/'tests.log', env)
    suites = ET.parse(root/'partition-mix-tests.xml').getroot().findall('testsuite')
    checks = {k: sum(int(s.attrib.get(k, 0)) for s in suites)
              for k in ('tests', 'failures', 'errors', 'skipped')}
    assert checks == dict(tests=17, failures=0, errors=0, skipped=0), checks
    profiles = {}
    for mixing in ('none', 'mhc'):
        target = root/(mixing+'-profile.json')
        execute([sys.executable, '-m', 'scripts.connection_partition_profile', '--readout', 'vocab_partition', '--partition-mixing', mixing,
                 '--output', str(target)], root/(mixing+'-profile.log'), env)
        profiles[mixing] = json.loads(target.read_text())
    ident = uuid.uuid4().hex[:8]
    out = root/'resume-partition'
    checkpoints = Path('../gate-checkpoints')
    cmd = [sys.executable, '-m', 'scripts.connection_train', '--variant', 'mhc', '--readout', 'vocab_partition', '--partition-mixing', 'mhc',
        '--connection-backend', 'nvidia', '--steps', '50', '--schedule-steps', '305176',
        '--microbatch', '16', '--attention-backend', 'sdpa_cudnn', '--matched-backbone-init',
        '--no-activation-checkpoint', '--no-fused-connection-grad', '--seed', '42',
        '--data-dir', '../data', '--output-dir', str(out), '--checkpoint-dir', str(checkpoints),
        '--run-id', ident, '--resume', 'auto']
    execute(cmd+['--segment-steps', '25'], root/'segment1.log', env, 85)
    progress = json.loads((out/'progress.json').read_text())
    assert progress['step'] == 25 and progress['run_id'] == ident
    execute(cmd, root/'segment2.log', env)
    trial = json.loads((out/'summary.json').read_text())
    config = json.loads((out/'config.json').read_text())
    rows = [json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows] == list(range(1, 51))
    assert all(math.isfinite(r['loss']) for r in rows)
    assert trial['complete'] and trial['steps'] == 50 and trial['training_tokens'] == 50*32768
    assert trial['run_id'] == ident and trial['source'] == source
    assert trial['final_dev_ce'] < trial['initial_dev_ce'], trial
    assert trial['peak_reserved_bytes'] < .9*22000*1024**2
    assert config['logging_backend'] == 'local' and config['residual_mixing'] == 'sinkhorn'
    assert config['applied_sinkhorn_iterations'] == 20 and config['model']['readout'] == 'vocab_partition' and config['model']['partition_mixing'] == 'mhc' and config['protocol']['schedule_steps'] == 305176
    assert (checkpoints/'latest.pt').exists() and (checkpoints/'best.pt').exists()
    assert not list(root.rglob('*.wandb')) and not (root/'wandb').exists()
    result = dict(complete=True, logging_backend='local', source=source, checks=checks,
        trial=trial, config=config, profiles=profiles, unique_training_steps=50,
        restart='Real exit85 at25, fresh process restores and completes50; exact optimizer/RNG unit check also passed',
        scope='Diagnostic weights remain in worker scratch; formal run starts independently at step0')
    (root/'partition-mix-gate.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
