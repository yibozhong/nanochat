"""Corrected independent attention write: structural tests, timing, restart."""
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import xml.etree.ElementTree as ET

from scripts.connection_native_gate import execute


def main():
    assert not os.environ.get('WANDB_API_KEY') and not Path('../.netrc').exists()
    root = Path('results')
    execute([sys.executable, '-m', 'pytest', 'tests/test_stream_attention.py',
        'tests/test_stream_independent_write.py',
        'tests/test_connection_training_resume.py::test_backbones_have_identical_initial_weights',
        'tests/test_mhc_nvidia_full.py::test_nvidia_model_no_deepseek_causality_cache_cce_and_layout',
        'tests/test_mhc_no_sinkhorn.py::test_full_model_causal_cache_cce_checkpoint',
        'tests/test_native_connections.py::test_full_size_backbone_initialization_and_optimizer_groups',
        '-v', '-s', '--junitxml=results/stream-tests.xml'], root/'tests.log')
    suites = ET.parse(root/'stream-tests.xml').getroot().findall('testsuite')
    checks = {k:sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    assert checks == dict(tests=36, failures=0, errors=0, skipped=0), checks
    modes = ('baseline', 'shared', 'per_stream_independent')
    trials = []
    for repetition in range(2):
        for mode in (modes if repetition == 0 else modes[::-1]):
            output = root/f'profile-{mode}-{repetition}.json'
            execute([sys.executable, '-m', 'scripts.connection_stream_profile', '--mode', mode,
                '--repetition', str(repetition), '--output', str(output)], output.with_suffix('.log'))
            trial = json.loads(output.read_text())
            assert trial['complete'] and trial['fits_22gb_budget'], trial
            trials.append(trial)
    assert len({r['gpu_uuid'] for r in trials}) == 1
    summary = {}
    for mode in modes:
        rows = [r for r in trials if r['mode'] == mode]
        seconds = statistics.mean(r['mean_seconds'] for r in rows)
        summary[mode] = dict(mean_seconds=seconds, tokens_per_second=32768/seconds,
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows), parameters=rows[0]['parameters'])
    new = summary['per_stream_independent']
    new['time_relative_to_shared'] = new['mean_seconds']/summary['shared']['mean_seconds']
    new['additional_peak_allocated_bytes'] = new['peak_allocated_bytes']-summary['shared']['peak_allocated_bytes']

    out = root/'restart'; checkpoints = Path('../gate-checkpoints')
    cmd = [sys.executable, '-m', 'scripts.connection_train', '--architecture', 'vanilla',
        '--variant', 'mhc', '--attention-mode', 'per_stream_independent',
        '--steps', '50', '--schedule-steps', '305176', '--microbatch', '16',
        '--attention-backend', 'sdpa_cudnn', '--matched-backbone-init',
        '--no-activation-checkpoint', '--no-fused-connection-grad', '--seed', '42',
        '--data-dir', '../data', '--output-dir', str(out), '--checkpoint-dir', str(checkpoints),
        '--run-id', 'independent-restart-check', '--resume', 'auto']
    execute(cmd+['--segment-steps', '25'], root/'restart-first.log', expected=85)
    first = json.loads((out/'progress.json').read_text()); assert first['step'] == 25
    execute(cmd, root/'restart-second.log')
    restart = json.loads((out/'summary.json').read_text())
    config = json.loads((out/'config.json').read_text())
    steps = [json.loads(s) for s in (out/'steps.jsonl').read_text().splitlines()]
    assert [s['step'] for s in steps] == list(range(1, 51))
    assert all(math.isfinite(s['loss']) for s in steps)
    assert restart['complete'] and restart['steps'] == 50 and restart['training_tokens'] == 50*32768
    assert config['model']['attention_mode'] == 'per_stream_independent'
    assert config['logging_backend'] == 'local' and config['protocol']['schedule_steps'] == 305176
    assert set(p.name for p in checkpoints.iterdir()) == {'latest.pt', 'best.pt'}
    shutil.rmtree(checkpoints)
    assert not list(root.rglob('*.pt')) and not list(root.rglob('*.wandb'))
    result = dict(complete=True, checks=checks, source=json.loads(Path('source.json').read_text()),
        logging_backend='local', formal_training=False, summary=summary, trials=trials,
        restart=dict(complete=True, first_step=25, final_step=50, summary=restart, config=config))
    (root/'stream-results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
