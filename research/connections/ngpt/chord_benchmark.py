"""Paired production-shape Chord timings on one GPU with the compiled scalar-beta control."""
import gc
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from xml.etree import ElementTree

import torch
import triton

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model
from nanochat.packed_training import PackedData, json_write, seed_all
from research.connections.ngpt.arc_residual_benchmark import MEASURE, TOKENS, WARMUP, time_steps, use_reference_writes

VARIANTS = {
    'single_step': dict(ngpt_velocity_mode='lookahead_transport'),
    'alpha_kappa': dict(ngpt_velocity_mode='lookahead_axis_write'),
    'split_fixed_beta': dict(ngpt_velocity_mode='lookahead_split_write'),
    'split_layer_beta_compiled': {},
    'split_layer_beta': {},
    'split_signed_alpha': dict(signed_chord_steps=True),
    'split_token_alpha': dict(token_chord_gates=True),
    'split_token_beta': dict(token_beta_gate=True),
    'deployed_token_beta': dict(token_beta_gate=True),
}
SUITES = ('chord_shared_writes', 'ngpt_lookahead', 'axis_write', 'split_write', 'split_learned_beta',
          'signed_chord', 'token_chord', 'token_beta_gate')


def check_writes():
    counts = {}
    for name in SUITES:
        xml = Path(f'results/{name}.xml')
        subprocess.run([sys.executable, '-m', 'pytest', '-q', f'tests/test_{name}.py',
                        f'--junitxml={xml}'], check=True)
        suites = list(ElementTree.parse(xml).getroot().iter('testsuite'))
        assert suites and all(int(s.get(k,0)) == 0 for s in suites for k in ('errors','failures','skipped'))
        counts[name] = sum(int(s.get('tests')) for s in suites)
        assert counts[name] > 0
    return counts


def benchmark(name, data):
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    seed_all(42)
    options = dict(ngpt_velocity_mode='lookahead_split_learned_beta')
    options.update(VARIANTS[name])
    config = ConnectionConfig(streams=1, residual_dynamics='ngpt_lookahead', velocity_init='token_position',
                              ngpt_force_rms_floor=1e-6, residual_projection_init='normal', **options)
    model = build_connection_model(config)
    if name == 'split_layer_beta_compiled':
        use_reference_writes(model)
    elif name == 'deployed_token_beta':
        last = model.blocks[-1].mlp_residual
        last.forward = last.reference_write
    model.prepare(compile_blocks=True)
    row = time_steps(model, data)
    assert all(math.isfinite(row[k]) for k in ('initial_ce','final_ce','median_ms'))
    assert row['final_ce'] < row['initial_ce'] and row['median_ms'] > 0
    return row


def main():
    gpu = torch.cuda.get_device_properties(0)
    assert torch.cuda.device_count() == 1 and gpu.major == 9 and 'MIG' not in gpu.name
    assert 'H100' in gpu.name or 'H200' in gpu.name
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    configure_attention('fa2')
    out = Path('results/comparison.json')
    report = dict(source=json.loads(Path('source.json').read_text()), complete=False, formal_training=False,
                  gpu=gpu.name, gpu_uuid=str(gpu.uuid), torch=torch.__version__, triton=triton.__version__,
                  warmup=WARMUP, measured=MEASURE, tokens_per_step=TOKENS, rounds={})
    json_write(out,report)
    report['tests'] = check_writes()
    json_write(out,report)
    data = PackedData('../data')
    for order in (list(VARIANTS), list(reversed(VARIANTS))):
        for name in order:
            row = benchmark(name,data)
            report['rounds'].setdefault(name,[]).append(row)
            json_write(out,report)
            print('CHORD_STEP_MS',name,row['median_ms'],flush=True)
    comparison = {}
    for name,rows in report['rounds'].items():
        medians = [r['median_ms'] for r in rows]
        ms = statistics.mean(medians)
        comparison[name] = dict(ms=ms,round_ms=medians,tokens_per_second=1000*TOKENS/ms,
                                peak_bytes=max(r['peak_bytes'] for r in rows))
    report.update(comparison=comparison,complete=True)
    json_write(out,report)
    print('CHORD_COMPARISON',json.dumps(comparison),flush=True)


if __name__ == '__main__':
    main()
