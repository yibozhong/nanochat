"""Gate a fresh ChordAttnRes run on correctness and same-card production-step timings."""
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import torch
import triton

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model
from nanochat.packed_training import PackedData, json_write, seed_all
from research.connections.ngpt.arc_residual_benchmark import MEASURE, WARMUP, TOKENS, time_steps


def main():
    output = Path('results/chord-attnres-profile.json')
    source = json.loads(Path('source.json').read_text())
    gpu = torch.cuda.get_device_properties(0)
    assert gpu.major == 9 and torch.cuda.device_count() == 1 and 'MIG' not in gpu.name
    if output.exists():
        prev = json.loads(output.read_text())
        assert prev['source'] == source
        if prev['complete']:
            print('Restored validated SM90 profile and fixed training backend:', prev['backend'], flush=True)
            return
    correction = source.get('chord_history_correction', False)
    report = dict(source=source, complete=False, formal_training=False, gpu=gpu.name, gpu_uuid=str(gpu.uuid),
                  torch=torch.__version__, triton=triton.__version__, warmup=WARMUP, measured=MEASURE,
                  tokens_per_step=TOKENS, rounds={k: [] for k in ('chord', 'direct', 'packed')})
    json_write(output, report)
    subprocess.run([sys.executable, '-m', 'pytest', '-xq', 'tests/test_chord_attnres.py',
                    *(['tests/test_chord_history_correction.py'] if correction else []),
                    '--junitxml=results/chord-attnres-tests.xml'], check=True)
    configure_attention('fa2')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    data = PackedData('../data')
    base = ConnectionConfig(**source['model'], streams=1, residual_dynamics='ngpt_lookahead',
                            velocity_init='token', ngpt_velocity_mode='lookahead_split_learned_beta',
                            chord_velocity_geometry='project_only', fuse_chord_read_norm=True,
                            ngpt_force_rms_floor=1e-6, attnres_block_size=source['attnres_block_size'])
    # Reverse the order for a second independent same-card timing of each kernel.
    for order in [('chord', 'packed', 'direct'), ('direct', 'packed', 'chord')]:
        for name in order:
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            seed_all(source['seed'])
            cfg = base if name == 'chord' else replace(base, chord_attnres=True, chord_attnres_backend=name,
                                                       chord_history_correction=correction)
            model = build_connection_model(cfg).prepare(compile_blocks=True)
            row = time_steps(model, data)
            del model
            assert all(math.isfinite(row[k]) for k in ('initial_ce','final_ce','median_ms'))
            assert row['final_ce'] < row['initial_ce'] and row['median_ms'] > 0
            extra = 12*2*512 + 14 if correction else 14*2*512
            assert row['parameters'] == 75_497_535 + (0 if name=='chord' else extra)
            report['rounds'][name].append(row)
            json_write(output, report)
            print('CHORD_ATTNRES_TIMING', name, row['median_ms'], row['peak_bytes'], flush=True)
    metrics = {name: dict(ms=statistics.mean(r['median_ms'] for r in rows),
                          peak_bytes=max(r['peak_bytes'] for r in rows))
               for name, rows in report['rounds'].items()}
    backend = min(('direct','packed'), key=lambda name: metrics[name]['ms'])
    overhead = metrics[backend]['ms']/metrics['chord']['ms'] - 1
    report.update(metrics=metrics, backend=backend, step_overhead=overhead)
    json_write(output, report)
    print('CHORD_ATTNRES_COMPARISON', json.dumps(metrics), 'backend', backend, flush=True)
    # Inspect the live allocation if depth mixing adds more than a quarter per step.
    if overhead > .25:
        raise RuntimeError(f'Best endpoint mix added {overhead:.1%} to a full Chord step; profile before formal training')
    report['complete'] = True
    json_write(output, report)


if __name__ == '__main__':
    main()
