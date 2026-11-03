"""Bounded stream-attention verification and two reversed timing rounds."""
import json
import os
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

from scripts.connection_native_gate import execute


def main():
    assert not os.environ.get('WANDB_API_KEY') and not Path('../.netrc').exists()
    root = Path('results')
    execute([sys.executable, '-m', 'pytest', 'tests/test_stream_attention.py',
        'tests/test_connection_training_resume.py::test_backbones_have_identical_initial_weights',
        'tests/test_mhc_nvidia_full.py::test_nvidia_model_no_deepseek_causality_cache_cce_and_layout',
        'tests/test_mhc_no_sinkhorn.py::test_full_model_causal_cache_cce_checkpoint',
        'tests/test_native_connections.py::test_full_size_backbone_initialization_and_optimizer_groups',
        '-v', '-s', '--junitxml=results/stream-tests.xml'], root/'tests.log')
    suites = ET.parse(root/'stream-tests.xml').getroot().findall('testsuite')
    checks = {k:sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    assert checks == dict(tests=17, failures=0, errors=0, skipped=0), checks
    trials = []
    for repetition in range(2):
        modes = ('baseline', 'shared', 'per_stream') if repetition==0 else ('per_stream', 'shared', 'baseline')
        for mode in modes:
            output = root/f'profile-{mode}-{repetition}.json'
            execute([sys.executable, '-m', 'scripts.connection_stream_profile', '--mode', mode,
                '--repetition', str(repetition), '--output', str(output)], output.with_suffix('.log'))
            trial = json.loads(output.read_text())
            assert trial['complete'] and trial['fits_22gb_budget'], trial
            trials.append(trial)
    assert len({t['gpu_uuid'] for t in trials}) == 1
    summary = {}
    for mode in ('baseline', 'shared', 'per_stream'):
        rows = [r for r in trials if r['mode']==mode]
        seconds = statistics.mean(r['mean_seconds'] for r in rows)
        summary[mode] = dict(mean_seconds=seconds, tokens_per_second=32768/seconds,
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows),
            parameters=rows[0]['parameters'])
    summary['per_stream']['time_relative_to_shared'] = summary['per_stream']['mean_seconds']/summary['shared']['mean_seconds']
    summary['per_stream']['additional_peak_allocated_bytes'] = summary['per_stream']['peak_allocated_bytes']-summary['shared']['peak_allocated_bytes']
    result = dict(complete=True, checks=checks, source=json.loads(Path('source.json').read_text()),
        logging_backend='local', formal_training=False, summary=summary, trials=trials)
    (root/'stream-results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
