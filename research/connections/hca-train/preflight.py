"""Validation and authorized training share the same GPU allocation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from policy import MODES, CORE_SOURCE, TEST_COUNTS, validate_probe, validate_shared


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(path)


def probe(mode, micro):
    path = Path('results/hca-startup') / f'{mode}-micro{micro}.json'
    subprocess.run([sys.executable, 'research/connections/hca-train/probe.py',
                    '--mode', mode, '--microbatch', str(micro), '--output', str(path)], check=True)
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=MODES, required=True); a = p.parse_args()
    import torch
    source = json.loads(Path('source.json').read_text())
    assert source['attention_mode'] == a.mode and source['formal_training'] is True
    assert torch.cuda.get_device_capability() == (9, 0)
    assert any(name in torch.cuda.get_device_name() for name in ('H100', 'H200'))
    assert torch.cuda.get_device_properties(0).total_memory >= 78000*1024**2
    if source.get('replay_debug_only', False):
        assert a.mode == 'fixed_hca'
        probe(a.mode, 8)
        # One reviewed diagnostic attempt at the failed production shape.
        # Even a coincidentally exact replay cannot unlock formal training.
        raise RuntimeError('Targeted replay diagnosis finished; inspect evidence before resuming the paired training plan')
    marker = Path('results/hca-startup/passed.json')
    if marker.exists():
        ready = json.loads(marker.read_text())
        validate_shared(ready, source['commit'])
        assert ready['run_id'] == source['run_id'] and ready['attention_mode'] == a.mode
        print('Reusing successful persistent HCA startup validation', flush=True)
        return
    if a.mode == 'fixed_hca':
        evidence = json.loads(Path('../core-evidence.json').read_text())
        assert evidence['source_commit'] == CORE_SOURCE and evidence['tests'] == TEST_COUNTS
        assert evidence['validated_bundle_commit'] == source['commit']
        subprocess.run([sys.executable, '-m', 'scripts.connection_hca_gate',
                        '--reuse-core-tests', '../core-evidence.json'], check=True)
        comparison = json.loads(Path('results/hca/validation.json').read_text())
        assert comparison['complete'] and comparison['tests'] == TEST_COUNTS
        assert comparison['source']['commit'] == source['commit']
        for micro in (16, 8, 4, 2, 1):
            reports = {}
            for mode in MODES:
                result = probe(mode, micro)
                if result['status'] in ('oom', 'over_budget'):
                    break
                validate_probe(result, mode, micro, source['commit'])
                reports[mode] = result
            if len(reports) == 2:
                break
        else:
            raise RuntimeError('No common microbatch fits the validated 78GB GPU budget')
        ready = dict(complete=True, formal_training=False, commit=source['commit'],
                     capability=list(torch.cuda.get_device_capability()), core_tests=TEST_COUNTS,
                     core_source=CORE_SOURCE, comparison_complete=True, microbatch=micro,
                     modes=reports, comparison_path='results/hca/validation.json')
        validate_shared(ready, source['commit'])
    else:
        ready = json.loads(Path('../validated-shared.json').read_text())
        micro = validate_shared(ready, source['commit'])
        # Same selected recipe, on this run's actual GPU allocation. Never
        # change its batch independently from the paired Fixed experiment.
        own = probe(a.mode, micro)
        validate_probe(own, a.mode, micro, source['commit'])
        write('results/hca-startup/own-allocation.json', own)
    ready.update(run_id=source['run_id'], attention_mode=a.mode)
    assert not any(Path('../hca-diagnostic-checkpoints').rglob('*.pt'))
    write(marker, ready)
    print(json.dumps(dict(event='HCA_STARTUP_PASSED', run_id=source['run_id'],
                          attention_mode=a.mode, report=ready), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
