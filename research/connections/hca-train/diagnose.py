"""Bounded, same-GPU root-cause investigation; never starts formal training."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

OUT = Path('results/hca-diagnosis')


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT/name; t = p.with_suffix('.tmp')
    t.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); t.replace(p)


@contextmanager
def stage_timers():
    import torch
    import nanochat.hca as hca
    rows, active = [], []
    counts = dict(pool_calls=0, pool_host_seconds=0., boundary_calls=0, boundary_host_seconds=0.)
    old_project, old_dense = hca.HCAAttention.project, hca.HCAAttention.dense
    old_attention, old_pool, old_boundary = hca.weighted_attention, hca.pool_step, hca.boundary_step
    def begin():
        e = torch.cuda.Event(enable_timing=True); e.record(); return e, time.perf_counter()
    def end(name, started):
        e = torch.cuda.Event(enable_timing=True); e.record()
        rows.append((name, started[0], e, time.perf_counter()-started[1]))
    def project(self, *args, **kwargs):
        start = begin(); y = old_project(self, *args, **kwargs); end('projections', start); return y
    def dense(self, *args, **kwargs):
        active.append(begin())
        try: return old_dense(self, *args, **kwargs)
        finally: active.pop()
    def attention(*args, **kwargs):
        if active: end('routing_pooling_and_mask_preparation', active[-1])
        start = begin(); y = old_attention(*args, **kwargs); end('weighted_attention', start); return y
    def pool(*args, **kwargs):
        start = time.perf_counter(); y = old_pool(*args, **kwargs)
        counts['pool_calls'] += 1; counts['pool_host_seconds'] += time.perf_counter()-start; return y
    def boundary(*args, **kwargs):
        start = time.perf_counter(); y = old_boundary(*args, **kwargs)
        counts['boundary_calls'] += 1; counts['boundary_host_seconds'] += time.perf_counter()-start; return y
    hca.HCAAttention.project, hca.HCAAttention.dense = project, dense
    hca.weighted_attention, hca.pool_step, hca.boundary_step = attention, pool, boundary
    try: yield rows, counts
    finally:
        hca.HCAAttention.project, hca.HCAAttention.dense = old_project, old_dense
        hca.weighted_attention, hca.pool_step, hca.boundary_step = old_attention, old_pool, old_boundary


def benchmark(mode):
    import torch
    from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
    from nanochat.connection_experiment import configure_attention
    from nanochat.packed_training import PackedData, seed_all, train_step
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    configure_attention('fa2'); seed_all(42)
    c = ConnectionConfig(variant='baseline', streams=1, attention_mode=mode)
    model = build_connection_model(c).prepare(compile_blocks=False)
    opt, wd = optimizer_for(model); data = PackedData('../data')
    times, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        schedule(opt, step, 305176, wd); torch.cuda.synchronize(); start = time.perf_counter()
        losses.append(train_step(model, opt, data, step, microbatch=8))
        torch.cuda.synchronize(); times.append(time.perf_counter()-start)
    timing = dict(mode=mode, formal_training=False, config=asdict(c), microbatch=8, batch_tokens=32768,
        gpu=torch.cuda.get_device_name(), gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        warmup_seconds=times[0], warm_step_seconds=times[1:], steady_step_seconds=sum(times[1:])/2,
        losses=losses, peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        backend='eager native FA2 baseline' if mode == 'full' else 'eager HCA reference',
        scope='Same GPU, model/data/global batch/microbatch; two warmed updates, not time-to-quality')
    write(mode+'-timing.json', timing)
    # One additional diagnostic update splits synchronized forward/backward/
    # optimizer wall time. Fine-grained events do not synchronize each token.
    schedule(opt, 3, 305176, wd); opt.zero_grad(set_to_none=True)
    parts = dict(data=0., forward=0., backward=0., optimizer=0.)
    with stage_timers() as (events, counts):
        for start_row in (48, 56):
            torch.cuda.synchronize(); start = time.perf_counter()
            x, y = data.batch('train', start_row, 8)
            torch.cuda.synchronize(); parts['data'] += time.perf_counter()-start
            start = time.perf_counter(); loss = model(x, y)
            torch.cuda.synchronize(); parts['forward'] += time.perf_counter()-start
            start = time.perf_counter(); (loss/2).backward()
            torch.cuda.synchronize(); parts['backward'] += time.perf_counter()-start
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        start = time.perf_counter(); opt.step(); torch.cuda.synchronize(); parts['optimizer'] = time.perf_counter()-start
    stages = {}
    for name, first, last, host in events:
        row = stages.setdefault(name, dict(calls=0, host_seconds=0., cuda_timeline_seconds=0.))
        row['calls'] += 1; row['host_seconds'] += host; row['cuda_timeline_seconds'] += first.elapsed_time(last)/1000
    timing.update(instrumented_step=parts, forward_stages=stages, recurrence=counts,
        timing_caution='CUDA timeline includes CPU launch gaps, not kernel-active time; instrumented step is separate from raw throughput')
    write(mode+'-timing.json', timing)
    print(json.dumps(dict(event='HCA_DIAG_TIMING', **timing), allow_nan=False), flush=True)


def pooling():
    import torch
    from nanochat.hca import pool_step
    torch.set_num_threads(2); torch.manual_seed(42)
    def sequential(k, v, score):
        pool = tuple(torch.zeros_like(k[:, 0], dtype=score.dtype) for _ in range(3))
        ks, vs = [], []
        for t in range(k.shape[1]):
            start = score.new_full((k.shape[0],), float(t % 16 == 0))
            pool, pk, pv = pool_step(k[:, t], v[:, t], score[:, t], pool, start)
            if (t+1) % 16 == 0: ks.append(pk); vs.append(pv)
        return torch.stack(ks, 1), torch.stack(vs, 1)
    def official_pattern(k, v, score):
        # Official Compressor.forward pattern: unflatten -> softmax over
        # chunk -> weighted sum. Use separate nanochat K/V and the existing
        # bounded score, zero position bias, to isolate implementation cost
        # without changing this comparison's pooling function.
        weight = (10*torch.tanh(score/10)).unflatten(1, (-1, 16)).softmax(dim=2)
        return tuple((weight*x.to(score.dtype).unflatten(1, (-1, 16))).sum(2) for x in (k, v))
    args = [torch.randn(2, 64, 2, 8, device='cuda', dtype=torch.float64, requires_grad=True) for _ in range(3)]
    a, b = sequential(*args), official_pattern(*args)
    upstream = [torch.randn_like(x) for x in a]
    for x, y in zip(a, b): torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad(a, args, upstream); gb = torch.autograd.grad(b, args, upstream)
    for x, y in zip(ga, gb): torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-12)
    result = dict(formal_training=False, complete=False, fp64_forward_vjp_equal=True,
        source='https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py',
        scope='Pooling only; completed fixed chunks, same function and inputs, no custom kernel or production change', modes={})
    for name, fn in [('current_sequential', sequential), ('official_vectorized_pattern', official_pattern)]:
        torch.manual_seed(123)
        raw = [torch.randn(8, 2048, 4, 128, device='cuda', dtype=d, requires_grad=True)
               for d in (torch.bfloat16, torch.bfloat16, torch.float32)]
        durations = []
        for _ in range(3):
            for x in raw: x.grad = None
            torch.cuda.synchronize(); start = time.perf_counter()
            outputs = fn(*raw); loss = sum(x.square().mean() for x in outputs); loss.backward()
            torch.cuda.synchronize(); durations.append(time.perf_counter()-start)
        result['modes'][name] = dict(forward_backward_seconds=durations, steady_seconds=sum(durations[1:])/2)
        del raw, outputs, loss
        torch.cuda.empty_cache()
    result['complete'] = True; write('pooling.json', result)
    print(json.dumps(dict(event='HCA_DIAG_POOL', **result)), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--worker', choices=('full', 'fixed_hca', 'dynamic_hca', 'pooling')); a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.worker:
        return pooling() if a.worker == 'pooling' else benchmark(a.worker)
    source = json.loads(Path('source.json').read_text())
    assert source['diagnostic_suite'] == 'hca_root_cause_v1'
    report = dict(event='HCA_ROOT_CAUSE_DIAGNOSIS', complete=False, formal_training=False,
                  source=source, comparisons={}, replay_checks={})
    for mode in ('full', 'fixed_hca', 'dynamic_hca', 'pooling'):
        subprocess.run([sys.executable, __file__, '--worker', mode], check=True)
        name = 'pooling.json' if mode == 'pooling' else mode+'-timing.json'
        report['comparisons'][mode] = json.loads((OUT/name).read_text()); write('diagnosis.json', report)
    for mode in ('fixed_hca', 'dynamic_hca'):
        path = OUT/(mode+'-controlled-replay.json')
        # A failed controlled replay is evidence, never a training pass.
        process = subprocess.run([sys.executable, 'research/connections/hca-train/replay_check.py',
            '--mode', mode, '--microbatch', '8', '--output', str(path)])
        report['replay_checks'][mode] = dict(exit_code=process.returncode,
            report=json.loads(path.read_text()) if path.exists() else None)
        write('diagnosis.json', report)
    report['complete'] = True; write('diagnosis.json', report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
