"""Measure terminal-only, preview-only and combined writes vs the current fused Chord."""
import gc
import json
from pathlib import Path
import math
import statistics
import subprocess
import sys

import torch
import triton

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model
from nanochat.packed_training import PackedData, json_write, seed_all
from research.connections.ngpt.arc_residual_benchmark import MEASURE, TOKENS, WARMUP, time_steps


def main():
    subprocess.run([sys.executable, '-m', 'pytest', '-xq', 'tests/test_chord_boundary.py',
                    'tests/test_chord_shared_writes.py', '--junitxml=results/checks.xml'], check=True)
    gpu = torch.cuda.get_device_properties(0)
    assert torch.cuda.device_count() == 1 and gpu.major == 9 and 'MIG' not in gpu.name
    configure_attention('fa2')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    modes = ('current', 'terminal', 'boundary', 'both')
    output = Path('results/comparison.json')
    report = dict(complete=False, source=json.loads(Path('source.json').read_text()), gpu=gpu.name,
                  gpu_uuid=str(gpu.uuid), torch=torch.__version__, triton=triton.__version__,
                  warmup=WARMUP, measured=MEASURE, tokens_per_step=TOKENS, rounds={m:[] for m in modes})
    json_write(output, report)
    data = PackedData('../data')
    cfg = ConnectionConfig(streams=1, residual_dynamics='ngpt_lookahead',
                           ngpt_velocity_mode='lookahead_split_learned_beta', ngpt_force_rms_floor=1e-6)
    for order in (modes, reversed(modes)):
        for mode in order:
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            seed_all(42)
            model = build_connection_model(cfg).prepare(compile_blocks=True, chord_kernel=mode)
            row = time_steps(model, data)
            del model
            assert math.isfinite(row['median_ms']) and row['final_ce'] < row['initial_ce']
            report['rounds'][mode].append(row)
            json_write(output, report)
            print('CHORD_STEP_MS',mode,row['median_ms'],flush=True)
    comparison = {}
    for mode, rows in report['rounds'].items():
        rounds = [r['median_ms'] for r in rows]
        ms = statistics.mean(rounds)
        comparison[mode] = dict(ms=ms, round_ms=rounds, tokens_per_second=TOKENS*1000/ms,
                                peak_bytes=max(r['peak_bytes'] for r in rows))
    report.update(complete=True, comparison=comparison)
    json_write(output, report)
    print('CHORD_BOUNDARY_COMPARISON',json.dumps(comparison),flush=True)


if __name__ == '__main__':
    main()
