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
        if expected is not None:
            assert process.returncode == expected, (name, process.returncode)
        return process.returncode

    run(['-c', 'from nanochat.connection_experiment import configure_attention; '
         'configure_attention('+repr(backend)+'); import pytest; '
         'raise SystemExit(pytest.main(["-v", "tests/test_state_tokens.py", "tests/test_state_attention.py", '
         '"--junitxml=results/state-tokens/tests.xml"]))'], 'tests')
    suites = ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts == dict(tests=26, failures=0, errors=0, skipped=0), counts
    run(['-m','scripts.connection_state_micro'], 'slot-backend-profile')
    profiles = {}
    attempts = []
    # Both backends use the same residual-Write architecture, initialization,
    # data, global batch and microbatch. Do not stop the second backend merely
    # because the first hits nonfinite training: preserve both failure records.
    for microbatch in (16, 8, 4):
        pair = {}
        for slots_backend in ('auto','triton'):
            name = 'state_tokens-'+slots_backend+'-micro'+str(microbatch)
            output = root/(name+'.json')
            run(['-m','scripts.connection_context_profile','--variant','state_tokens','--sequence-length','2048',
                 '--microbatch',str(microbatch),'--attention-backend',backend,
                 '--state-attention-backend',slots_backend,
                 '--output',str(output)], 'profile-'+name, expected=None)
            result = json.loads(output.read_text()); attempts.append(result)
            pair[slots_backend] = result
        if any(r['status']=='oom' for r in pair.values()): continue
        if any(r.get('peak_reserved_bytes',0) >= .9*r['device_total_bytes'] for r in pair.values()): continue
        profiles['state_tokens'] = pair['triton']
        profiles['state_tokens_torch'] = pair['auto']
        break
    assert 'state_tokens' in profiles, attempts
    run(['-m','scripts.connection_context_profile','--variant','baseline','--sequence-length','2048',
         '--microbatch',str(microbatch),'--attention-backend',backend,
         '--output',str(root/'baseline.json')], 'profile-baseline')
    profiles['baseline'] = json.loads((root/'baseline.json').read_text())
    (root/'profile-summary.json').write_text(json.dumps(dict(profiles=profiles, attempts=attempts),indent=2)+'\n')
    for result in profiles.values():
        assert result['complete'] and result['causality_passed'], result
        assert profiles['baseline']['input_sha256'] == result['input_sha256']
        assert profiles['baseline']['gpu_uuid'] == result['gpu_uuid']

    args = ['-m','scripts.connection_train','--variant','state_tokens','--matched-backbone-init',
            '--steps','50','--schedule-steps','305176','--microbatch',str(microbatch),'--data-dir','../data',
            '--state-attention-backend','triton',
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
    assert config['model']['state_write_residual'] and config['model']['state_attention_backend']=='triton'
    assert summary['final_dev_ce'] < summary['initial_dev_ce']
    import torch
    state = torch.load('../state-gate-checkpoints/latest.pt', map_location='cpu', weights_only=False)
    assert all(torch.isfinite(p).all() for p in state['model'].values())
    assert not list(Path('results').rglob('*.wandb'))
    report = dict(complete=True, formal_training=False, checks=counts, profiles=profiles, restart=summary,
        profile_attempts=attempts, microbatch=microbatch,
        slot_backend_profile=json.loads(Path('results/state-slot-micro.json').read_text()),
        source=json.loads(Path('source.json').read_text()),
        time_ratio=profiles['state_tokens']['mean_seconds']/profiles['baseline']['mean_seconds'],
        triton_vs_torch_time_ratio=profiles['state_tokens']['mean_seconds']/profiles['state_tokens_torch']['mean_seconds'],
        additional_peak_allocated_bytes=profiles['state_tokens']['peak_allocated_bytes']-profiles['baseline']['peak_allocated_bytes'],
        limitations='One GPU/seed.60-step profiles and25->50 resume validate implementation, not long-run quality or stability.')
    (root/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('STATE TOKEN GATE PASSED', flush=True)


if __name__ == '__main__': main()
