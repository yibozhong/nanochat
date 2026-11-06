"""Bounded correctness, same-GPU cost, and real resume checks; no formal run."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    root = Path('results/state-tokens'); root.mkdir(parents=True, exist_ok=True)
    backend = os.environ['STATE_ATTENTION_BACKEND']
    def run(args, name, expected=0):
        print('BEGIN '+name, flush=True)
        with (root/(name+'.log')).open('w') as log:
            process = subprocess.Popen([sys.executable, *args], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line); print(line, end='', flush=True)
            process.wait()
        assert process.returncode == expected, (name, process.returncode)

    run(['-c', 'from nanochat.connection_experiment import configure_attention; '
         'configure_attention('+repr(backend)+'); import pytest; '
         'raise SystemExit(pytest.main(["-v", "tests/test_state_tokens.py", '
         '"--junitxml=results/state-tokens/tests.xml"]))'], 'tests')
    suites = ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts == dict(tests=8, failures=0, errors=0, skipped=0), counts
    profiles = {}
    for v in ('baseline', 'state_tokens'):
        run(['-m','scripts.connection_context_profile','--variant',v,'--sequence-length','2048',
             '--microbatch','16','--attention-backend',backend,
             '--output',str(root/(v+'.json'))], 'profile-'+v)
        result = json.loads((root/(v+'.json')).read_text())
        assert result['complete'] and result['causality_passed'], result
        assert result['peak_reserved_bytes'] < .9*result['device_total_bytes'], result
        profiles[v] = result
    assert profiles['baseline']['input_sha256'] == profiles['state_tokens']['input_sha256']
    assert profiles['baseline']['gpu_uuid'] == profiles['state_tokens']['gpu_uuid']

    args = ['-m','scripts.connection_train','--variant','state_tokens','--matched-backbone-init',
            '--steps','50','--schedule-steps','305176','--microbatch','16','--data-dir','../data',
            '--attention-backend',backend,'--output-dir',str(root/'restart'),
            '--checkpoint-dir','../state-gate-checkpoints','--run-id','state-token-check',
            '--segment-seconds','3600','--segment-steps','25']
    run(args, 'restart-first', 85)
    run(args+['--resume','auto','--segment-steps','0'], 'restart-second')
    summary = json.loads((root/'restart/summary.json').read_text())
    config = json.loads((root/'restart/config.json').read_text())
    rows = [json.loads(line) for line in (root/'restart/steps.jsonl').read_text().splitlines()]
    assert summary['complete'] and summary['steps'] == 50
    assert [r['step'] for r in rows] == list(range(1, 51))
    assert all(math.isfinite(r['loss']) for r in rows)
    assert config['logging_backend'] == 'local' and config['applied_sinkhorn_iterations'] == 0
    assert summary['final_dev_ce'] < summary['initial_dev_ce']
    import torch
    state = torch.load('../state-gate-checkpoints/latest.pt', map_location='cpu', weights_only=False)
    assert all(torch.isfinite(p).all() for p in state['model'].values())
    assert not list(Path('results').rglob('*.wandb'))
    report = dict(complete=True, formal_training=False, checks=counts, profiles=profiles, restart=summary,
        source=json.loads(Path('source.json').read_text()),
        time_ratio=profiles['state_tokens']['mean_seconds']/profiles['baseline']['mean_seconds'],
        additional_peak_allocated_bytes=profiles['state_tokens']['peak_allocated_bytes']-profiles['baseline']['peak_allocated_bytes'],
        limitations='One GPU/seed.60-step profiles and25->50 resume validate implementation, not long-run quality or stability.')
    (root/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('STATE TOKEN GATE PASSED', flush=True)


if __name__ == '__main__': main()
