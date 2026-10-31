"""Bounded integration gate; no formal training and no W&B runs."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET


def execute(cmd, log, expected=0):
    with log.open('w') as handle:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end='', flush=True)
            handle.write(line); handle.flush()
        code = proc.wait()
    if code != expected:
        raise RuntimeError(f'{cmd[2:4]} returned {code}, expected {expected}; see {log}')


def main():
    root = Path('results')
    source = json.loads(Path('source.json').read_text())
    execute([sys.executable, '-m', 'pytest', 'tests/test_native_connections.py',
             'tests/test_connection_training_resume.py', '-v', '-s',
             '--junitxml=results/native-tests.xml'], root/'tests.log')
    suites = ET.parse(root/'native-tests.xml').getroot().findall('testsuite')
    checks = {k: sum(int(s.attrib.get(k, 0)) for s in suites)
              for k in ('tests', 'failures', 'errors', 'skipped')}
    assert checks == dict(tests=11, failures=0, errors=0, skipped=0), checks
    trials = {}
    for variant in ('baseline', 'mhc', 'mhc_no_sinkhorn'):
        ident = uuid.uuid4().hex[:8]
        out = root/('native-'+variant)
        checkpoints = Path('../gate-checkpoints')/variant
        cmd = [sys.executable, '-m', 'scripts.connection_train', '--architecture', 'native',
            '--variant', variant, '--steps', '10', '--schedule-steps', '305176',
            '--microbatch', '16', '--attention-backend', 'auto', '--matched-backbone-init',
            '--no-activation-checkpoint', '--no-fused-connection-grad', '--seed', '42',
            '--data-dir', '../data', '--output-dir', str(out), '--checkpoint-dir', str(checkpoints),
            '--run-id', ident, '--resume', 'auto']
        execute(cmd+['--segment-steps', '5'], root/(variant+'-segment1.log'), 85)
        progress = json.loads((out/'progress.json').read_text())
        assert progress['step'] == 5 and progress['run_id'] == ident
        execute(cmd, root/(variant+'-segment2.log'))
        result = json.loads((out/'summary.json').read_text())
        config = json.loads((out/'config.json').read_text())
        rows = [json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows] == list(range(1, 11))
        assert all(math.isfinite(r['loss']) for r in rows)
        assert result['complete'] and result['training_tokens'] == 10*32768
        assert result['run_id'] == ident and result['source'] == source
        assert config['logging_backend'] == 'local' and config['model']['architecture'] == 'native'
        assert config['model']['window_pattern'] == 'SSSL'
        assert result['final_dev_ce'] < result['initial_dev_ce'], result
        assert set(p.name for p in checkpoints.iterdir()) == {'latest.pt', 'best.pt'}
        trials[variant] = dict(summary=result, config=config)
        shutil.rmtree(checkpoints)  # diagnostic weights never leave worker scratch
    assert not list(root.rglob('*.wandb')) and not (root/'wandb').exists()
    result = dict(complete=True, source=source, checks=checks, trials=trials,
        logging_backend='local', diagnostic_tokens=3*10*32768,
        scope='Implementation checks; no quality or throughput comparison, no formal training',
        restart='Each native variant: exit85 at5, fresh process restores model/optimizer/RNG/data and completes10')
    (root/'native-gate.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
