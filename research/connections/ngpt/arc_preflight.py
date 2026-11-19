"""Same-allocation checks for Arc before an independent seed-42 formal process."""
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import torch


def main():
    begin = time.perf_counter()
    source = json.loads(Path('source.json').read_text())
    assert source['matrix_optimizer'] == 'arc'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    gpu = torch.cuda.get_device_name()
    major, minor = torch.cuda.get_device_capability()
    if source.get('hopper_only'):
        assert (major, minor) == (9, 0) and any(name in gpu for name in ('H100', 'H200')) and 'MIG' not in gpu, gpu
    marker = Path(f'results/arc-sm{major}{minor}.json')
    if marker.exists():
        previous = json.loads(marker.read_text())
        assert previous['complete'] and previous['source'] == source and previous['torch'] == torch.__version__
        return
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_arc.py',
                    '--junitxml=results/arc-tests.xml'], check=True)
    root = Path(tempfile.mkdtemp(prefix='arc-diagnostic-', dir='..')).resolve()
    results = {}
    for name in ('muon', 'arc'):
        out = root/name
        steps = 64 if name == 'arc' else 32
        cmd = [sys.executable, '-m', 'scripts.connection_train', '--variant', 'baseline',
               '--streams', '1', '--matrix-optimizer', name, '--arc-mu', str(source['arc_mu']),
               '--arc-beta', str(source['arc_beta']), '--residual-projection-init', 'normal',
               '--ngpt-force-rms-floor', '1e-6', '--steps', str(steps), '--schedule-steps', '305176',
               '--attention-backend', 'fa2', '--microbatch', '16', '--data-dir', '../data',
               '--output-dir', str(out), '--seed', '42', '--run-id', 'arc-local-debug']
        if name == 'arc':
            first = subprocess.run([*cmd, '--segment-steps', '32'])
            if first.returncode != 85:
                raise RuntimeError(f'Expected a durable restart at 32, got {first.returncode}')
            progress = json.loads((out/'progress.json').read_text())
            assert progress['step'] == 32 and progress['tokens'] == 32*32768
            cmd.extend(['--resume', 'auto'])
        subprocess.run(cmd, check=True)
        info = json.loads((out/'summary.json').read_text())
        assert info['complete'] and info['steps'] == steps and info['dev_tokens'] == 131072
        assert abs(info['initial_dev_ce'] - source['reference_initial_dev_ce']) < 1e-3
        assert math.isfinite(info['final_dev_ce']) and info['final_dev_ce'] < info['initial_dev_ce']
        rows = [json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines()]
        assert [row['step'] for row in rows] == list(range(1, steps+1))
        assert all(math.isfinite(row['loss']) and row['seconds'] > 0 for row in rows)
        # Same data and warm-up interval on this GPU, before the Arc checkpoint.
        results[name] = dict(initial_dev_ce=info['initial_dev_ce'], final_dev_ce=info['final_dev_ce'],
                             median_step_seconds=statistics.median(r['seconds'] for r in rows[16:32]),
                             peak_allocated_bytes=info['peak_allocated_bytes'])
        if name == 'arc':
            events = [json.loads(line)['values'] for line in (out/'diagnostics/arc-local-debug/events.jsonl').read_text().splitlines()]
            last = next(e for e in reversed(events) if e.get('step') == steps and 'dev/ce' in e)
            assert 0 < last['optimizer/mean_query_degrees'] < 90
            assert last['optimizer/max_relative_radius_error'] < 1e-5
            results[name].update({k:v for k,v in last.items() if k.startswith('optimizer/')})
        shutil.rmtree(out)
    root.rmdir()
    result = dict(complete=True, source=source, formal_training=False, gpu=gpu, torch=torch.__version__,
                  cuda=torch.version.cuda, models=results, resumed_at_step=32,
                  arc_to_muon_time=results['arc']['median_step_seconds']/results['muon']['median_step_seconds'],
                  evaluation_wall_seconds=time.perf_counter()-begin)
    marker.write_text(json.dumps(result, indent=2)+'\n')
    print('ARC_PREFLIGHT_PASSED', json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
