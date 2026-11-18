"""Compare additive lookahead compilation boundaries on one GPU, after the Arc gate."""
from dataclasses import replace
from functools import partial
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
from nanochat.lookahead import LookaheadModel
from nanochat.packed_training import PackedData, json_write, seed_all, train_step
from research.connections.ngpt.lookahead_read_triton import fused_read

REPORT = Path('results/lookahead-compiler.json')
OPTIONS = dict(dynamic=False, fullgraph=True, options={'emulate_precision_casts': True})
MODES = ('current', 'full', 'attention_eager', 'ffn_eager', 'compiled_read', 'triton_read',
         'triton_branches')


def read(h, v, mu):
    return (h.float() + mu[0] * v).to(h.dtype)


def sublayer(block, attention, h, v, cos, sin, cache):
    site = block.attn_residual if attention else block.mlp_residual
    q = read(h, v, site.mu)
    force = block.attention_branch(q, cos, sin, cache) if attention else block.mlp_branch(q)
    return site(h, v, force)


def install(model, mode):
    if mode == 'current':
        return
    if mode == 'full':
        model.compiled_steps = [torch.compile(b.momentum_forward, **OPTIONS) for b in model.blocks]
        return
    def make_step(b):
        # Split only the selected branch. The opposite whole sublayer retains fusion.
        attn_update = torch.compile(b.attn_residual.forward, **OPTIONS)
        ffn_update = torch.compile(b.mlp_residual.forward, **OPTIONS)
        query = fused_read if mode.startswith('triton_') else torch.compile(read, **OPTIONS)
        attn_branch = torch.compile(b.attention_branch, **OPTIONS) if mode == 'triton_branches' else b.attention_branch
        ffn_branch = torch.compile(b.mlp_branch, **OPTIONS) if mode == 'triton_branches' else b.mlp_branch
        attn = torch.compile(partial(sublayer, b, True), **OPTIONS) if mode == 'ffn_eager' else None
        ffn = torch.compile(partial(sublayer, b, False), **OPTIONS) if mode == 'attention_eager' else None
        def step(h, v, cos, sin, cache=None):
            if attn is None:
                q = query(h, v, b.attn_residual.mu)
                h, v = attn_update(h, v, attn_branch(q, cos, sin, cache))
            else:
                h, v = attn(h, v, cos, sin, cache)
            if ffn is not None:
                return ffn(h, v, cos, sin, cache)
            return ffn_update(h, v, ffn_branch(query(h, v, b.mlp_residual.mu)))
        return step
    model.compiled_steps = [make_step(b) for b in model.blocks]


def pytest_configure(config):
    # Loaded only with -p in this diagnostic's pytest child.
    mode = os.environ['LOOKAHEAD_DIAGNOSTIC_MODE']
    original = LookaheadModel.prepare
    def prepare(model, *args, **kwargs):
        value = original(model, *args, **kwargs)
        if model.compiled_steps is not None and model.config.residual_mode == 'lookahead_only':
            install(model, mode)
        return value
    LookaheadModel.prepare = prepare


def tiny(checkpoint=False):
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='lookahead_only', lookahead_mu_init=.3,
                            ngpt_force_rms_floor=1e-6, activation_checkpoint=checkpoint)


def relative(a, b):
    return ((a.double()-b.double()).norm() / a.double().norm().clamp_min(1e-10)).item()


def parity(mode, seed, checkpoint):
    # Match test_compiled_checkpoint_forward_backward, including RNG consumed by both builds.
    seed_all(seed)
    c = tiny(checkpoint)
    ref = build_connection_model(replace(c, activation_checkpoint=False)).prepare()
    target = build_connection_model(c).prepare(compile_blocks=True)
    install(target, mode)
    target.load_state_dict(ref.state_dict())
    x = torch.randint(0, 256, (2,16), device='cuda')
    y = (x + 3) % 256
    a = ref(x, y, dense=True)
    b = target(x, y, dense=True)
    a.backward()
    b.backward()
    errors = {}
    mus = {}
    for (name, p), (key, q) in zip(ref.named_parameters(),target.named_parameters(),strict=True):
        assert key == name
        errors[name] = relative(p.grad, q.grad)
        if name.endswith('.mu'):
            mus[name] = dict(ref=p.grad.item(), actual=q.grad.item(),
                             abs_error=abs(p.grad.item()-q.grad.item()), relative=errors[name])
    valid_loss = bool(torch.isclose(a, b, rtol=2e-4, atol=2e-4))
    return dict(seed=seed, checkpoint=checkpoint, valid=valid_loss and all(e < .04 for e in errors.values()),
                loss=[a.item(),b.item()], worst=sorted(errors.items(), key=lambda kv: -kv[1])[:4], mus=mus)


def read_checks():
    rows = []
    for n in (509, 4096, 32769):
        seed_all(42)
        h = torch.randn(n, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(n, device='cuda', requires_grad=True)
        upstream = torch.randn_like(h)
        for mu_value in (0., .3, -2.):
            mu = torch.tensor([mu_value], device='cuda', requires_grad=True)
            eager = read(h, v, mu)
            custom = fused_read(h, v, mu)
            a = torch.autograd.grad(eager, (h,v,mu), upstream)
            b = torch.autograd.grad(custom, (h,v,mu), upstream)
            ref = (v.double() * upstream.double()).sum()
            scale = (v.double() * upstream.double()).abs().sum().item()
            err = abs(ref.item()-b[2].item())
            rows.append(dict(n=n, mu=mu_value, pass_check=torch.equal(eager, custom)
                             and torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                             and err <= 2e-7 * max(scale, 1e-10),
                             dmu_abs_error_to_fp64=err, dmu_eager=a[2].item(),
                             dmu_triton=b[2].item(), sum_abs_products=scale))
    return rows


def local_vjp():
    # Same primal inputs and upstream vector for each local VJP. Inspect forward
    # drift separately from dmu amplification when the dot product cancels.
    seed_all(42)
    model = build_connection_model(tiny()).prepare()
    build_connection_model(tiny()).prepare()
    x = torch.randint(0,256,(2,16),device='cuda')
    y = (x+3)%256
    site = model.blocks[1].attn_residual
    saved = {}
    original = site.query
    def tapped(h, v):
        q = original(h,v)
        saved.update(h=h.detach(), v=v.detach(), q=q)
        q.retain_grad()
        return q
    site.query = tapped
    model(x,y,dense=True).backward()
    h,v,q = saved['h'],saved['v'],saved['q']
    dq = q.grad.to(h.dtype)
    terms = (v.double()*dq.double())
    report = {'eager_site_grad':site.mu.grad.item(), 'fp64_dot':terms.sum().item(),
              'sum_abs_products':terms.abs().sum().item(),
              'cancellation_ratio':(terms.abs().sum()/terms.sum().abs().clamp_min(1e-30)).item()}
    def local(fn, args, upstream=None):
        base = [t.detach().requires_grad_() for t in args]
        eager = fn(*base)
        actual = torch.compile(fn, **OPTIONS)(*base)
        if upstream is None:
            upstream = torch.randn_like(eager)
        ref = torch.autograd.grad(eager,base,upstream)
        got = torch.autograd.grad(actual,base,upstream)
        return dict(forward_relative=relative(eager,actual), input_vjp_relative=[relative(a,b) for a,b in zip(ref,got)])
    block = model.blocks[1]
    cos,sin = model.cos[:,:x.size(1)],model.sin[:,:x.size(1)]
    report['read_fixed_vjp'] = local(read, (h,v,site.mu), dq)
    report['attention_fixed_vjp'] = local(lambda z:block.attention_branch(z,cos,sin), (q.detach().to(h.dtype),))
    report['ffn_fixed_vjp'] = local(block.mlp_branch, (q.detach().to(h.dtype),))
    return report


def benchmark(mode):
    seed_all(42)
    c = ConnectionConfig(streams=1, residual_dynamics='lookahead_only', ngpt_force_rms_floor=1e-6)
    model = build_connection_model(c).prepare(compile_blocks=True)
    install(model, mode)
    opt, wd = optimizer_for(model)
    data = PackedData('../data')
    times = []
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(32):
        schedule(opt,step,305176,wd)
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = train_step(model,opt,data,step,microbatch=16)
        torch.cuda.synchronize()
        if step >= 16:
            times.append(time.perf_counter()-start)
        losses.append(loss)
    return dict(median_ms=1000*statistics.median(times), times=times,
                first_ce=losses[0], final_ce=losses[-1], peak_bytes=torch.cuda.max_memory_allocated())


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    configure_attention('fa2')
    info = dict(complete=False, formal_training=False, device=torch.cuda.get_device_name(),
                device_uuid=str(torch.cuda.get_device_properties(0).uuid),torch=torch.__version__,
                cuda=torch.version.cuda, parity={}, suites={}, timing={})
    def persist():
        json_write(REPORT,info)
    persist()
    info['triton_unit'] = read_checks()
    info['local'] = local_vjp()
    persist()
    for mode in MODES:
        torch._dynamo.reset()
        try:
            checks = [parity(mode,42,False), parity(mode,42,True), parity(mode,137,False)]
        except Exception as exc:
            if mode == 'current':
                raise
            checks = [dict(valid=False, error=repr(exc))]
        info['parity'][mode] = checks
        print('LOOKAHEAD_PARITY',mode,[x['valid'] for x in checks],flush=True)
        persist()
    # The historical full graph can serve as a speed-only control if it fails parity.
    eligible = [m for m in MODES if all(row['valid'] for row in info['parity'][m])
                and (not m.startswith('triton_') or all(row['pass_check'] for row in info['triton_unit']))]
    for mode in eligible:
        result = subprocess.run([sys.executable,'-m','pytest','-q','-p',__spec__.name,
                                 'tests/test_lookahead_only.py','--junitxml=results/lookahead-'+mode+'.xml'],
                                env={**os.environ,'LOOKAHEAD_DIAGNOSTIC_MODE':mode})
        info['suites'][mode] = result.returncode
        persist()
    valid = [m for m in eligible if info['suites'][m]==0]
    if 'current' not in valid:
        raise RuntimeError('The reference implementation failed the existing checks')
    timed = list(valid)
    if 'full' not in timed and not any('error' in row for row in info['parity']['full']):
        timed.append('full')
    info['speed_only'] = [m for m in timed if m not in valid]
    for order in (timed, list(reversed(timed))):
        for mode in order:
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            row = benchmark(mode)
            info['timing'].setdefault(mode, []).append(row)
            print('LOOKAHEAD_TIMING',mode,row['median_ms'],flush=True)
            persist()
    reference = statistics.mean(r['median_ms'] for r in info['timing']['current'])
    info['qualified_time_ratios_to_current'] = {
        m: statistics.mean(r['median_ms'] for r in info['timing'][m]) / reference for m in valid}
    info['complete'] = True
    persist()


if __name__=='__main__':
    main()
