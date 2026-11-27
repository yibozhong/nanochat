"""Same-GPU, same-data comparison using the ordinary training entry point."""
import json
import math
from pathlib import Path
import statistics
import subprocess
import tempfile

from .run_queue import ROOT, read, train_args
from .submit import BATCH_TOKENS, save


def compare(source, directory, env):
    observations = {}
    for method in ('standard', 'sequential'):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='backward-comparison-') as tmp:
            out = Path(tmp)
            flags = train_args(source)
            if method == 'standard':
                flags.remove('--sequential-backward')
            args = [*flags, '--output-dir', tmp, '--run-id', 'diagnostic-' + method, '--steps', '50']
            subprocess.run(args, env=env, check=True, timeout=2400)
            summary = read(out/'summary.json')
            rows = [json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines()]
            assert summary['complete'] and len(rows) == summary['steps'] == 50
            assert all(math.isfinite(row['loss']) for row in rows)
            durations = [row['seconds'] for row in rows[10:]]
            obs = {k: summary[k] for k in ('gpu', 'training_tokens', 'dev_tokens', 'initial_dev_ce',
                                             'final_dev_ce', 'estimated_training_flops', 'peak_allocated_bytes')}
            obs.update(timed_steps=len(durations), tokens_per_second=len(durations)*BATCH_TOKENS/sum(durations),
                       median_step_seconds=statistics.median(durations))
            if method == 'sequential':
                obs['at_standard_flop_budget'] = read(out/'equal-compute.json')
            observations[method] = obs
    base, seq = observations['standard'], observations['sequential']
    assert all(base[k] == seq[k] for k in ('gpu', 'training_tokens', 'dev_tokens'))
    assert abs(base['initial_dev_ce'] - seq['initial_dev_ce']) < 1e-6
    point = seq['at_standard_flop_budget']
    assert 0 <= base['estimated_training_flops'] - point['training_flops'] < seq['estimated_training_flops'] / 50
    observations['delta_dev_ce_equal_tokens'] = seq['final_dev_ce'] - base['final_dev_ce']
    observations['delta_dev_ce_at_standard_flop_budget'] = point['dev_ce'] - base['final_dev_ce']
    observations['throughput_ratio'] = seq['tokens_per_second'] / base['tokens_per_second']
    save(directory/'sequential-comparison.json', observations)
    print('BACKWARD_COMPARISON', json.dumps(observations), flush=True)
