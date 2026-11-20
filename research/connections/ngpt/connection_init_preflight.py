"""Check the selected random-init connector on the training GPU before starting from zero."""
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


def main():
    source = json.loads(Path('source.json').read_text())
    variant = source['connection_variant']
    assert variant in ('mhc', 'attnres') and source['hopper_only']
    marker = Path('results/connection-init-sm90.json')
    if marker.exists():
        previous = json.loads(marker.read_text())
        assert previous['complete'] and previous['source'] == source
        return

    test = f'tests/test_nonzero_residual.py::test_matched_connection_initialization[{variant}]'
    subprocess.run([sys.executable, '-m', 'pytest', '-q', test], check=True)
    with tempfile.TemporaryDirectory(prefix='connection-init-', dir='..') as work:
        out = Path(work).resolve()
        args = [sys.executable, '-m', 'scripts.connection_train', *sys.argv[1:],
                '--steps', '32', '--output-dir', str(out), '--run-id', f'{variant}-init-check',
                '--segment-seconds', '3600']
        first = subprocess.run([*args, '--segment-steps', '12'])
        assert first.returncode == 85, first.returncode
        progress = json.loads((out/'progress.json').read_text())
        assert progress['step'] == 12 and progress['run_id'] == f'{variant}-init-check'
        subprocess.run([*args, '--resume', 'auto'], check=True)
        summary = json.loads((out/'summary.json').read_text())
        config = json.loads((out/'config.json').read_text())
        rows = [json.loads(row) for row in (out/'steps.jsonl').read_text().splitlines()]
        assert [row['step'] for row in rows] == list(range(1, 33))
        assert all(math.isfinite(row['loss']) for row in rows)
        assert summary['complete'] and summary['source'] == source and summary['variant'] == variant
        assert math.isfinite(summary['final_dev_ce']) and summary['final_dev_ce'] < summary['initial_dev_ce']
        assert config['logging_backend'] == 'local' and config['actual_attention_backend'] == 'fa2'
        assert config['gpu_capability'] == [9, 0] and config['model']['variant'] == variant
        assert config['model']['residual_projection_init'] == 'normal' and config['protocol']['matched_backbone_init']
        report = dict(complete=True, source=source, gpu=config['gpu'], resume_step=12,
                      median_step_seconds=statistics.median(row['seconds'] for row in rows[-12:]),
                      peak_allocated_bytes=summary['peak_allocated_bytes'],
                      steps=32, initial_dev_ce=summary['initial_dev_ce'], final_dev_ce=summary['final_dev_ce'])
    marker.write_text(json.dumps(report, indent=2)+'\n')
    print(f"{variant} initialization and 12-to-32-step resume passed", flush=True)


if __name__ == '__main__':
    main()
