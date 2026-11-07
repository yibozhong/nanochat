"""Isolated fixed-hash FFN benchmark. No production integration or NTP run."""
import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import triton as tr
import triton.language as tl
import deep_gemm as dg
from transformer_engine.pytorch import GroupedLinear
from transformer_engine.pytorch.module.grouped_linear import is_module_grouped_tensor_path_supported
import transformer_engine_torch as tex
from nanochat.optim import MuonAdamW


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


@tr.jit
def _gather(X, I, Y, ROWS: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = p // D, p % D
    index = tl.load(I + row, row < ROWS, -1)
    value = tl.load(X + index * D + col, (row < ROWS) & (index >= 0), 0)
    tl.store(Y + p, value, row < ROWS)


def gather(x, indices):
    out = torch.empty((indices.numel(), x.shape[1]), device=x.device, dtype=x.dtype)
    _gather[(tr.cdiv(out.numel(), 1024),)](x, indices, out, out.shape[0], out.shape[1], 1024)
    return out


class Permute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, forward_indices, backward_indices):
        ctx.save_for_backward(backward_indices)
        return gather(x.contiguous(), forward_indices)

    @staticmethod
    def backward(ctx, grad):
        return gather(grad.contiguous(), ctx.saved_tensors[0]), None, None


class Plan:
    def __init__(self, assignment, experts, alignment):
        started = time.perf_counter()
        self.experts = experts
        self.n = len(assignment)
        self.counts = np.bincount(assignment, minlength=experts).astype(np.int64)
        self.sizes = (((self.counts + alignment - 1) // alignment) * alignment).tolist()
        order = np.argsort(assignment, kind='stable')
        pack = np.full(sum(self.sizes), -1, dtype=np.int64)
        unpack = np.empty(self.n, dtype=np.int64)
        offset = original = 0
        for count, padded in zip(self.counts, self.sizes):
            rows = order[original:original + count]
            pack[offset:offset + count] = rows
            unpack[rows] = np.arange(offset, offset + count)
            offset += padded
            original += count
        assert np.array_equal(pack[unpack], np.arange(self.n))
        self.metadata_cpu_ms = (time.perf_counter() - started) * 1000
        self.pack_index = torch.from_numpy(pack).cuda()
        self.unpack_index = torch.from_numpy(unpack).cuda()
        self.layout = torch.from_numpy(np.repeat(np.arange(experts, dtype=np.int32), self.sizes)).cuda()
        self.sizes_gpu = torch.tensor(self.sizes, device='cuda', dtype=torch.int32)
        self.sizes_cpu = torch.tensor(self.sizes, dtype=torch.int64)
        self.rows = len(pack)

    def pack(self, x):
        return Permute.apply(x, self.pack_index, self.unpack_index)

    def unpack(self, x):
        return Permute.apply(x, self.unpack_index, self.pack_index)

    def report(self):
        return dict(tokens=self.n, padded_tokens=self.rows, padding_ratio=self.rows/self.n,
                    empty_experts=int((self.counts == 0).sum()), min_tokens=int(self.counts.min()),
                    max_tokens=int(self.counts.max()), median_tokens=float(np.median(self.counts)),
                    counts=self.counts.tolist(), metadata_cpu_ms=self.metadata_cpu_ms)


class DGLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, master, plan, *weights):
        # Parameters are disjoint views of master. Cast the bank once, not E times.
        w = master.to(torch.bfloat16)
        y = torch.empty((x.shape[0], w.shape[1]), device='cuda', dtype=x.dtype)
        dg.m_grouped_bf16_gemm_nt_contiguous(x, w, y, plan.layout)
        ctx.save_for_backward(x, w)
        ctx.plan = plan
        return y

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        grad = grad.contiguous()
        dx = torch.empty_like(x)
        dg.m_grouped_bf16_gemm_nn_contiguous(grad, w, dx, ctx.plan.layout)
        # This version requires an FP32 accumulator. Zero also defines empty-group grads.
        dw = torch.zeros(w.shape, device='cuda', dtype=torch.float32)
        dg.k_grouped_bf16_gemm_tn_contiguous(grad, x, dw, ctx.plan.sizes,
                                          ctx.plan.sizes_gpu, dw)
        return (dx, None, None, *dw.unbind(0))


@torch.compile(dynamic=True, fullgraph=True)
def activation(x):
    return F.relu(x).square()


class ExpertFFN(nn.Module):
    def __init__(self, backend, experts, width, hidden, seed=42):
        super().__init__()
        self.backend = backend
        self.experts = experts
        self.native_grouped = backend == 'te' and is_module_grouped_tensor_path_supported(None, torch.bfloat16)
        # Keep all expert matrices logically independent for the existing Muon optimizer.
        generator = torch.Generator(device='cuda').manual_seed(seed)
        self.up_bank = torch.randn(experts, hidden, width, device='cuda', generator=generator) / math.sqrt(width)
        self.down_bank = torch.randn(experts, width, hidden, device='cuda', generator=generator) / math.sqrt(hidden)
        if backend == 'te':
            self.up = GroupedLinear(experts, width, hidden, bias=False, params_dtype=torch.float32,
                                    device='cuda', use_grouped_tensor=self.native_grouped)
            self.down = GroupedLinear(experts, hidden, width, bias=False, params_dtype=torch.float32,
                                      device='cuda', use_grouped_tensor=self.native_grouped)
            for i in range(experts):
                getattr(self.up, f'weight{i}').data = self.up_bank[i]
                getattr(self.down, f'weight{i}').data = self.down_bank[i]
        else:
            self.up = nn.ParameterList([nn.Parameter(self.up_bank[i]) for i in range(experts)])
            self.down = nn.ParameterList([nn.Parameter(self.down_bank[i]) for i in range(experts)])

    def ordered_params(self):
        if self.backend == 'te':
            return ([getattr(self.up, f'weight{i}') for i in range(self.experts)] +
                    [getattr(self.down, f'weight{i}') for i in range(self.experts)])
        return list(self.up) + list(self.down)

    def packed(self, x, plan):
        if self.backend == 'dg':
            h = DGLinear.apply(x, self.up_bank, plan, *self.up)
            return DGLinear.apply(activation(h), self.down_bank, plan, *self.down)
        if self.backend == 'te':
            splits = plan.sizes_gpu.to(torch.int64) if self.native_grouped else plan.sizes_cpu
            # TE requires AMP for BF16 activations with FP32 master parameters.
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                return self.down(activation(self.up(x, splits)), splits)
        # Reference only (and dense E=1 anchor). Weight casts match nanochat Linear.
        outputs = []
        for i, part in enumerate(x.split(plan.sizes)):
            h = F.linear(part, self.up[i].to(x.dtype))
            outputs.append(F.linear(activation(h), self.down[i].to(x.dtype)))
        return torch.cat(outputs)

    def forward(self, x, plan):
        if self.backend == 'torch' and self.experts == 1:
            return self.packed(x, plan)
        return plan.unpack(self.packed(plan.pack(x), plan))

    def optimizer(self):
        params = self.ordered_params()
        groups = []
        for shape in sorted({p.shape for p in params}):
            groups.append(dict(kind='muon', params=[p for p in params if p.shape == shape],
                               lr=.005, momentum=.95, ns_steps=5, beta2=.9, weight_decay=0.))
        return MuonAdamW(groups)


def compare(actual, expected):
    a, b = actual.float(), expected.float()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    error = (a-b).norm().item()
    scale = b.norm().item()
    ratio = error / max(scale, 1e-7)
    assert error <= .02 * scale + 1e-7, (error, scale, ratio)
    return ratio


def correctness(outdir, backends=('dg', 'te')):
    reports = []
    for n, e, d, h, magnitude in [(257, 4, 64, 256, 1.), (19, 8, 64, 256, 1.),
                                 (129, 4, 64, 256, 0.), (129, 4, 64, 256, 1e-4),
                                 (257, 4, 512, 2048, 1.)]:
        assignment = np.arange(n) % (e if n != 19 else 3)
        generator = torch.Generator(device='cuda').manual_seed(1337)
        x = (torch.randn(n, d, device='cuda', generator=generator) * magnitude).bfloat16()
        dy = torch.randn(n, d, device='cuda', generator=generator).bfloat16()
        reference = ExpertFFN('torch', e, d, h)
        rx = x.detach().requires_grad_()
        ry = reference(rx, Plan(assignment, e, 1))
        rg = torch.autograd.grad(ry, [rx] + reference.ordered_params(), dy)
        expected = [ry.detach().cpu()] + [v.detach().cpu() for v in rg]
        del ry, rg, reference, rx
        for backend in backends:
            model = ExpertFFN(backend, e, d, h)
            tx = x.detach().requires_grad_()
            y = model(tx, Plan(assignment, e, 128 if backend == 'dg' else 1))
            grads = torch.autograd.grad(y, [tx] + model.ordered_params(), dy)
            ratios = [compare(a, b.cuda()) for a, b in zip([y] + list(grads), expected)]
            reports.append(dict(backend=backend, tokens=n, experts=e, width=d, hidden=h,
                                magnitude=magnitude, max_relative_l2=max(ratios),
                                compared_tensors=len(ratios), passed=True))
            save(outdir/'checks.json', reports)
            print('CHECK', json.dumps(reports[-1]), flush=True)
            del model, tx, y, grads
        del x, dy, expected
        gc.collect(); torch.cuda.empty_cache()
    return reports


def timed(fn, iterations=10, warmup=3):
    started = time.perf_counter()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter()-started
    begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    started = time.perf_counter()
    begin.record()
    for _ in range(iterations):
        fn()
    end.record(); end.synchronize()
    return dict(wall_ms=(time.perf_counter()-started)*1000/iterations,
                cuda_stream_ms=begin.elapsed_time(end)/iterations,
                iterations=iterations, warmup=warmup, setup_warmup_seconds=warmup_seconds)


def profile_one(case, backend, assignment, repetition, outdir):
    n, d, h, e = case['tokens'], case['width'], case['hidden'], case['experts']
    torch.cuda.reset_peak_memory_stats()
    plan = Plan(assignment, e, 128 if backend == 'dg' else 1)
    model = ExpertFFN(backend, e, d, h)
    generator = torch.Generator(device='cuda').manual_seed(2026)
    x = torch.randn(n, d, device='cuda', dtype=torch.bfloat16, generator=generator).requires_grad_()
    dy = (torch.randn(n, d, device='cuda', dtype=torch.bfloat16, generator=generator) / math.sqrt(n)).contiguous()
    def zero():
        model.zero_grad(set_to_none=True); x.grad = None
    def whole():
        zero(); model(x, plan).backward(dy)
    measurements = {}
    # Warm actual full-shape kernels; check forward and all gradients before timing.
    whole(); torch.cuda.synchronize()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.ordered_params())
    assert torch.isfinite(x.grad).all()
    measurements['complete_fwd_bwd'] = timed(whole)
    xp = plan.pack(x).detach().requires_grad_()
    dyp = plan.pack(dy).detach()
    measurements['packed_forward'] = timed(lambda: model.packed(xp, plan))
    yp = model.packed(xp, plan)
    def backward_only():
        model.zero_grad(set_to_none=True); xp.grad = None
        yp.backward(dyp, retain_graph=True)
    measurements['packed_backward'] = timed(backward_only)
    del yp, xp, dyp
    def reorder():
        x.grad = None
        plan.unpack(plan.pack(x)).backward(dy)
    measurements['pack_unpack_fwd_bwd'] = timed(reorder)
    # Exercise the real optimizer on independent 2D expert parameters.
    whole()
    opt = model.optimizer()
    measurements['optimizer_only'] = timed(opt.step)
    def update():
        whole(); opt.step()
    measurements['complete_with_muon'] = timed(update)
    assert all(torch.isfinite(p).all() and torch.isfinite(p.grad).all() for p in model.ordered_params())
    # Trace one representative shape outside all timing windows.
    trace_summary = None
    if d == 512 and e == 16 and case['routing'] == 'uniform' and repetition == 0:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            whole()
            torch.cuda.synchronize()
        trace = outdir/f'{backend}-trace.json'
        prof.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())['traceEvents']
        kernels = [v for v in events if v.get('cat') == 'kernel']
        trace_summary = dict(kernel_count=len(kernels), names=dict(Counter(v['name'] for v in kernels)))
        save(outdir/f'{backend}-trace-summary.json', trace_summary)
        trace.unlink()  # Only small summaries leave the worker.
    result = dict(case=case, backend=backend, repetition=repetition, complete=True,
                  routing=plan.report(), te_native_grouped=model.native_grouped,
                  measurements=measurements, trace=trace_summary,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  input_vjp_and_parameter_gradients_finite=True)
    del model, opt, x, dy, plan
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-backend', choices=('dg', 'te'))
    args = parser.parse_args()
    outdir = args.output; outdir.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0)
    assert importlib.metadata.version('transformer-engine') == '2.20.2'
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.recompile_limit = 128
    torch._dynamo.config.accumulated_recompile_limit = 1024
    # The container exposes CUDA even when the optional nvidia-smi CLI is absent.
    try:
        smi = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip()
        smi_error = None
    except (OSError, subprocess.CalledProcessError) as error:
        smi, smi_error = None, repr(error)
    probe_path = Path('results/device-probe.json')
    device = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  deepgemm=getattr(dg, '__version__', 'unknown'), te=importlib.metadata.version('transformer-engine'),
                  cublaslt=tex.get_cublasLt_version(),
                  te_native_grouped_available=is_module_grouped_tensor_path_supported(None, torch.bfloat16),
                  nvidia_smi=smi, nvidia_smi_unavailable=smi_error,
                  device_probe=json.loads(probe_path.read_text()) if probe_path.exists() else None)
    save(outdir/'device.json', device)
    if args.check_backend:
        try:
            correctness(outdir, (args.check_backend,))
        except Exception as error:
            save(outdir/'check-failure.json', dict(backend=args.check_backend,
                 error=repr(error), traceback=traceback.format_exc()))
            raise
        return
    # Independent CUDA processes preserve the other backend's evidence on failure.
    checks, check_failures, passed_backends = [], [], set()
    for backend in ('te', 'dg'):
        check_dir = outdir/f'checks-{backend}'
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                  '--data', str(args.data), '--output', str(check_dir),
                                  '--check-backend', backend], check=False)
        checks_path = check_dir/'checks.json'
        backend_checks = json.loads(checks_path.read_text()) if checks_path.exists() else []
        checks.extend(backend_checks)
        if process.returncode == 0 and len(backend_checks) == 5 and all(c['passed'] for c in backend_checks):
            passed_backends.add(backend)
        else:
            failure_path = check_dir/'check-failure.json'
            failure = json.loads(failure_path.read_text()) if failure_path.exists() else {}
            check_failures.append(dict(stage='correctness', backend=backend,
                                       returncode=process.returncode, detail=failure))
        save(outdir/'checks.json', checks)
        save(outdir/'check-status.json', dict(passed_backends=sorted(passed_backends),
                                            failures=check_failures))
    data = np.memmap(args.data, dtype=np.uint16, mode='r').reshape(-1, 2049)
    tokens = np.array(data[:16, :-1]).reshape(-1).astype(np.int64)
    assert len(tokens) == 32768
    token_hash = hashlib.sha256(tokens.tobytes()).hexdigest()
    # Token identity always maps to the same bucket. No learned router or target IDs.
    lookup = np.random.default_rng(42).permutation(32768)
    cases = []
    for d, h, expert_counts in [(512, 2048, [1, 4, 16, 64, 256]), (64, 256, [256, 1024, 4096])]:
        for e in expert_counts:
            for routing in (['uniform'] if e == 1 else ['uniform', 'fixed_token_hash']):
                cases.append(dict(tokens=32768, width=d, hidden=h, experts=e, routing=routing))
    report = dict(complete=False, formal_training=False, device=device, checks=checks,
                  source=json.loads(Path('source.json').read_text()), token_sha256=token_hash,
                  scope='Single FFN and Muon operator benchmark; no NTP/full-model speed or quality claim',
                  results=[], failures=check_failures)
    save(outdir/'report.json', report)
    for index, case in enumerate(cases):
        assignment = (np.arange(32768) % case['experts'] if case['routing'] == 'uniform'
                      else lookup[tokens] % case['experts'])
        order = ['torch', 'torch'] if case['experts'] == 1 else ['dg', 'te', 'te', 'dg']
        order = [backend for backend in order if backend == 'torch' or backend in passed_backends]
        counts = Counter()
        failed = set()
        for backend in order:
            if backend in failed:
                continue
            try:
                gc.collect(); torch.cuda.empty_cache()
                result = profile_one(case, backend, assignment, counts[backend], outdir)
                counts[backend] += 1
                report['results'].append(result)
                print('RESULT', json.dumps(dict(case=case, backend=backend,
                      wall_ms=result['measurements']['complete_with_muon']['wall_ms'],
                      fwbw_ms=result['measurements']['complete_fwd_bwd']['wall_ms'],
                      memory=result['peak_allocated_bytes'])), flush=True)
            except Exception as error:
                failed.add(backend)
                report['failures'].append(dict(case=case, backend=backend, error=repr(error), traceback=traceback.format_exc()))
                print('CASE_FAILED', case, backend, repr(error), flush=True)
                if 'illegal memory' in str(error).lower() or 'device-side assert' in str(error).lower():
                    save(outdir/'report.json', report)
                    raise
            save(outdir/'report.json', report)
    report['complete'] = not report['failures']
    report['finished_all_cases'] = True
    save(outdir/'report.json', report)
    print('BENCHMARK_FINISHED', json.dumps(dict(complete=report['complete'], results=len(report['results']), failures=len(report['failures']))), flush=True)
    if report['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
