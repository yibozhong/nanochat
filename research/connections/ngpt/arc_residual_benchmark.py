"""Single-device production-shape Arc Residual ablations, including the compiled reference."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import torch
import triton

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
from nanochat.lookahead import LookaheadModel
from nanochat.momentum import normalize_residual, tangent_project, inverse_softplus
from nanochat.packed_training import PackedData, json_write, seed_all, train_step
from research.connections.ngpt.arc_residual_triton import compile_arc_step, fused_write

MODES = {'vanilla': 'vanilla', 'ngpt': 'ngpt', 'arc': 'ngpt_lookahead',
         'arc_triton': 'ngpt_lookahead', 'lookahead_split': 'lookahead_only',
         'lookahead_10b': 'lookahead_only'}
REPORT = Path('results/comparison.json')
WARMUP, MEASURE, TOKENS = 16, 32, 32768


def install(model):
    model.compiled_steps = [compile_arc_step(b) for b in model.blocks]


def pytest_configure(config):
    # Only the explicitly loaded pytest child installs the candidate.
    original = LookaheadModel.prepare

    def prepare(model, *args, **kwargs):
        original(model, *args, **kwargs)
        if model.compiled_steps is not None and model.config.residual_mode == 'ngpt_lookahead':
            install(model)
        return model

    LookaheadModel.prepare = prepare


def local_checks():
    from nanochat.lookahead import LookaheadResidual

    seed_all()
    checks = []
    for width, case in [(128, 'ordinary'), (512, 'ordinary'), (512, 'force_floor'),
                        (512, 'antipodal'), (128, 'blend_floor')]:
        site = LookaheadResidual(ConnectionConfig(streams=1, residual_dynamics='ngpt_lookahead',
                                                 ngpt_force_rms_floor=1e-6)).cuda()
        h = normalize_residual(torch.randn(23, width, device='cuda'))
        force = torch.randn_like(h).bfloat16()
        if case == 'force_floor':
            force[:12].zero_()
        if case in ('antipodal', 'blend_floor'):
            h[:12].fill_(1.)
            force[:12] = -h[:12].bfloat16()
            with torch.no_grad():
                site.raw.fill_(inverse_softplus(1. if case == 'antipodal' else .5))
        v = tangent_project(h, torch.randn_like(h))
        inputs = [x.requires_grad_() for x in (h, v, force)] + [site.raw]
        reference = site(h, v, force)
        candidate = fused_write(h, v, force, site.alpha(), site.decay, site.rms_floor)
        for exp, actual in zip(reference, candidate, strict=True):
            torch.testing.assert_close(actual, exp, atol=3e-5, rtol=3e-5)
        upstream = tuple(torch.randn_like(x) for x in reference)
        exp = torch.autograd.grad(reference, inputs, upstream)
        actual = torch.autograd.grad(candidate, inputs, upstream)
        errors = {}
        for name, a, b in zip(('h', 'v', 'force', 'raw_alpha'), exp, actual, strict=True):
            err = ((b.float()-a.float()).double().norm()/a.double().norm().clamp_min(1e-7)).item()
            assert err < (.006 if a.dtype == torch.bfloat16 else .001), (width, case, name, err)
            errors[name] = err
        checks.append(dict(width=width, case=case, gradient_errors=errors))
    return checks


def regression_suite():
    path = 'results/arc-residual-tests.xml'
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_ngpt_lookahead.py',
                           '-p', 'research.connections.ngpt.arc_residual_benchmark', '--junitxml='+path])
    suites = list(ET.parse(path).getroot().iter('testsuite'))
    counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert proc.returncode == 0 and counts == dict(tests=8, failures=0, errors=0, skipped=0), counts
    return counts


def write_timings():
    from nanochat.lookahead import LookaheadResidual

    config = ConnectionConfig(streams=1, residual_dynamics='ngpt_lookahead', ngpt_force_rms_floor=1e-6)
    site = LookaheadResidual(config).cuda()
    seed_all()
    h = normalize_residual(torch.randn(TOKENS, 512, device='cuda'))
    v = tangent_project(h, torch.randn_like(h))
    force = torch.randn_like(h).bfloat16()
    for x in (h, v, force):
        x.requires_grad_()
    upstream = (torch.randn_like(h), torch.randn_like(v))

    def candidate(h, v, force):
        return fused_write(h, v, force, site.alpha(), site.decay, site.rms_floor)

    timings = {}
    for name, fn in [('compiled', site.forward), ('triton', candidate)]:
        fn = torch.compile(fn, fullgraph=True, dynamic=False, options={'emulate_precision_casts': True})
        samples = []
        for i in range(30):
            begin, middle, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            begin.record()
            outputs = fn(h, v, force)
            middle.record()
            torch.autograd.grad(outputs, (h, v, force, site.raw), upstream)
            end.record()
            end.synchronize()
            if i >= 10:
                samples.append((begin.elapsed_time(middle), middle.elapsed_time(end)))
        timings[name] = dict(forward_ms=statistics.median(x[0] for x in samples),
                             backward_ms=statistics.median(x[1] for x in samples))
    return timings


def benchmark(mode, data):
    seed_all()
    config = ConnectionConfig(streams=1, residual_dynamics=MODES[mode],
                              residual_projection_init='normal', ngpt_force_rms_floor=1e-6)
    model = build_connection_model(config)
    if mode == 'lookahead_split':
        model.prepare(compile_blocks=True, compile_lookahead_branches=True)
    else:
        model.prepare(compile_blocks=True)
        if mode == 'arc_triton':
            install(model)
    optimizer, wd = optimizer_for(model)
    durations, losses = [], []
    for i in range(WARMUP + MEASURE):
        if i == WARMUP:
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        schedule(optimizer, i, 305176, wd)
        loss = train_step(model, optimizer, data, i, microbatch=16, batch_tokens=TOKENS)
        torch.cuda.synchronize()
        if i >= WARMUP:
            durations.append(1000 * (time.perf_counter() - start))
        losses.append(loss)
    return dict(config=asdict(config), parameters=sum(p.numel() for p in model.parameters()),
                median_ms=statistics.median(durations), milliseconds=durations,
                peak_bytes=torch.cuda.max_memory_allocated(), initial_ce=losses[0], final_ce=losses[-1])


def summarize(runs):
    medians = {mode: statistics.mean(x['median_ms'] for x in rows) for mode, rows in runs.items()}
    return {mode: dict(ms=ms, tokens_per_second=TOKENS*1000/ms, step_overhead_pct=100*(ms/medians['vanilla']-1),
                       order_medians_ms=[r['median_ms'] for r in runs[mode]],
                       max_peak_bytes=max(r['peak_bytes'] for r in runs[mode]))
            for mode, ms in medians.items()}


def main():
    assert torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    assert props.major == 9 and ('H100' in props.name or 'H200' in props.name)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    configure_attention('fa2')
    info = dict(complete=False, formal_training=False, gpu=props.name, gpu_uuid=str(props.uuid),
                torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__,
                source=json.loads(Path('source.json').read_text()), warmup=WARMUP, measured=MEASURE,
                tokens=TOKENS, suite={}, rounds={}, errors={})
    json_write(REPORT, info)
    try:
        info['local_checks'] = local_checks()
        json_write(REPORT, info)
        info['suite'] = regression_suite()
        json_write(REPORT, info)
        info['write_only_gpu_ms'] = write_timings()
    except Exception:
        info['errors']['arc_triton'] = traceback.format_exc()
        print(info['errors']['arc_triton'], flush=True)
    json_write(REPORT, info)
    data = PackedData('../data')
    for modes in (list(MODES), list(reversed(MODES))):
        for mode in modes:
            if mode in info['errors']:
                continue
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            print('BENCHMARK_START', mode, flush=True)
            try:
                row = benchmark(mode, data)
            except Exception:
                info['errors'][mode] = traceback.format_exc()
                print(info['errors'][mode], flush=True)
            else:
                info['rounds'].setdefault(mode, []).append(row)
                print('BENCHMARK_MEDIAN_MS', mode, row['median_ms'], flush=True)
            json_write(REPORT, info)
    paired = {mode: rows for mode, rows in info['rounds'].items() if len(rows) == 2}
    if 'vanilla' in paired:
        info['comparison'] = summarize(paired)
    if {'arc', 'arc_triton'} <= paired.keys():
        before, after = (info['comparison'][m]['ms'] for m in ('arc', 'arc_triton'))
        info['arc_step_time_reduction_pct'] = 100*(1-after/before)
    info['complete'] = not info['errors'] and paired.keys() == MODES.keys()
    json_write(REPORT, info)
    print('FINAL_COMPARISON', json.dumps(info.get('comparison')), flush=True)
    if not info['complete']:
        raise RuntimeError('Incomplete comparison; inspect the per-mode errors')


if __name__ == '__main__':
    main()
