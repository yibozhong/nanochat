"""Production-shape comparison within one debug allocation; no W&B."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--label', required=True)
p.add_argument('--kernel-file')
p.add_argument('--mode', choices=['full', 'fixed_hca', 'dynamic_hca'], default='fixed_hca')
p.add_argument('--block-size', type=int, default=16)
p.add_argument('--steps', type=int, default=20)
p.add_argument('--projection', choices=['native','tf32x3','bf16x3'])
p.add_argument('--rotary',choices=['native','fused'])
a = p.parse_args()
assert os.environ['WANDB_MODE'] == 'disabled'
if a.kernel_file:
    spec = importlib.util.spec_from_file_location('nanochat.hca_triton', a.kernel_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

import torch
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
from nanochat.connection_experiment import configure_attention
from nanochat.packed_training import PackedData, seed_all, train_step

if a.rotary:
    import nanochat.hca as hca
    hca.USE_FUSED_ROTARY = a.rotary == 'fused'

if a.projection:
    import nanochat.hca_projection as projection
    if a.projection == 'native':
        projection.score_linear = lambda x,w: torch.nn.functional.linear(x.float(),w)
    else:
        projection.PRECISION = a.projection

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
configure_attention('fa2')
seed_all(42)
c = ConnectionConfig(variant='baseline', streams=1, attention_mode=a.mode,
                     hca_backend='triton', hca_block_size=a.block_size)
model = build_connection_model(c).prepare(compile_blocks=False)
opt, wd = optimizer_for(model)
data = PackedData('../data')
seconds, losses = [], []
torch.cuda.reset_peak_memory_stats()
for step in range(5 + a.steps):
    schedule(opt, step, 305176, wd)
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss = train_step(model, opt, data, step, microbatch=16)
    torch.cuda.synchronize()
    if step >= 5:
        seconds.append(time.perf_counter() - start)
        losses.append(loss)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
    schedule(opt, 5 + a.steps, 305176, wd)
    train_step(model, opt, data, 5 + a.steps, microbatch=16)
    torch.cuda.synchronize()
kernels = {}
for e in prof.events():
    if str(e.device_type).endswith('CUDA'):
        row = kernels.setdefault(e.name, dict(calls=0, microseconds=0.))
        row['calls'] += 1
        row['microseconds'] += e.device_time_total
report = dict(args=vars(a), formal_training=False, gpu=torch.cuda.get_device_name(),
              gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
              source=json.loads(Path('source.json').read_text()), seconds=seconds,
              median_seconds=statistics.median(seconds), mean_seconds=statistics.mean(seconds),
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              peak_reserved_bytes=torch.cuda.max_memory_reserved(),
              losses=losses, kernels=[dict(name=k, **v) for k, v in kernels.items()],
              note='CUDA profiler events may include annotations; do not sum as wall time.')
out = Path('results/optimization'); out.mkdir(parents=True, exist_ok=True)
(out / (a.label + '.json')).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(label=a.label, seconds=report['median_seconds'], losses=losses)), flush=True)
