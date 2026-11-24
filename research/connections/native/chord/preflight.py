"""One-allocation native Chord gate. Runs locally on the GPU, without W&B."""
import argparse
import gc
import math
import statistics
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from nanochat.chord_triton import fused_split_write
from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
from nanochat.engine import KVCache
from nanochat.packed_training import PackedData, json_write, restore_rng, seed_all, snapshot, train_step


def formula(h, v, force, p):
    def inner(x, y):
        return (x*y).mean(-1, keepdim=True)

    def unit(x):
        return x * inner(x, x).clamp_min(1e-12).rsqrt()

    perp, parallel, a, b, c = p.unbind()
    delta = unit(force) - h
    par = v * (inner(v, delta) / inner(v, v).clamp_min(1e-12))
    n = unit(h + perp*(delta - par) + parallel*par)
    chord = n - h
    u = v - n * inner(v, n)
    d = chord - n * inner(chord, n)
    along = u * (inner(u, d) / inner(u, u).clamp_min(1e-12))
    return n, a*u + b*along + c*(d-along)


def kernel_checks():
    for rest in (False, True):
        h, v, f = torch.randn(3, 2, 19, 512, device='cuda').unbind()
        h = h * h.square().mean(-1, keepdim=True).rsqrt()
        v = torch.zeros_like(v) if rest else v - (v*h).mean(-1, keepdim=True)*h
        p = torch.tensor([.09, .13, .62, -.2, 1.1], device='cuda')
        inputs = [x.detach().requires_grad_() for x in (h, v, f, p)]
        actual = fused_split_write(*inputs, 1e-6, 'project_only')
        expected = formula(*inputs)
        upstream = tuple(torch.randn_like(x) for x in actual)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
        for x, y in zip(torch.autograd.grad(actual, inputs, upstream),
                        torch.autograd.grad(expected, inputs, upstream)):
            torch.testing.assert_close(x, y, atol=4e-4, rtol=3e-4)


@torch.no_grad()
def paired_and_causal(model, tied, ids):
    model.eval()
    tied.eval()
    h = model.hidden(ids)
    baseline_h = tied.hidden(ids)
    if model.config.native_chord_write == 'token_vnorm':
        # The tiled token read and scalar read have different FP32 reduction trees.
        torch.testing.assert_close(h, baseline_h, rtol=.01, atol=.016)
        assert (h.float()-baseline_h.float()).norm() < .002 * baseline_h.float().norm()
    else:
        torch.testing.assert_close(h, baseline_h, rtol=0, atol=0)
    # Activate lookahead and anisotropic writes, plus native smear, for causality and KV.
    model.native.smear_lambda.fill_(.2)
    for layer in model.chord.layers:
        for site in (layer.attention, layer.ffn):
            site.mu.fill_(.3)
            if layer.token_controls:
                site.token_weight.normal_(std=.03/model.config.width**.5)
            if site.writes is not None:
                site.writes.copy_(site.writes.new_tensor([.35, .8]))
    h = model.hidden(ids)
    altered = ids.clone()
    altered[:, 17:] = ids[:, 17:].roll(1, 1)
    torch.testing.assert_close(model.hidden(altered)[:, :17], h[:, :17], rtol=0, atol=0)
    c = model.config
    cache = KVCache(1, c.heads, ids.size(1), c.width//c.heads, c.depth, 'cuda', torch.bfloat16)
    prefill = model.hidden(ids[:, :-1], cache)
    decoded = model.hidden(ids[:, -1:], cache)
    assert cache.get_pos() == ids.size(1)
    torch.testing.assert_close(torch.cat((prefill, decoded), dim=1), h, rtol=.02, atol=.035)


def make(c, compile_blocks):
    seed_all(42)
    return build_connection_model(c).prepare(compile_blocks=compile_blocks)


def speed_and_resume(c, data, folder, *, check_resume, steps):
    model = make(c, True)
    opt, wd = optimizer_for(model)
    times = []
    for step in range(steps):
        schedule(opt, step, 305176, wd)
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = train_step(model, opt, data, step, microbatch=16)
        torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
    assert math.isfinite(loss)
    if check_resume:
        active = [s for layer in model.chord.layers for s in (layer.attention, layer.ffn) if s.writes is not None]
        assert len(active) == 2*c.depth-1
        assert all(s.writes.grad is not None and bool((s.writes.grad != 0).all()) for s in active)
        assert all(s.raw_beta.grad is not None and s.raw_beta.grad.abs().item() > 0 for s in active)
        if c.native_chord_write == 'token_vnorm':
            for layer in model.chord.layers:
                for site in (layer.attention, layer.ffn):
                    grad = site.token_weight.grad
                    assert grad is not None and (grad.float().norm(dim=-1) > 0).all()
        path = folder/'local.pt'
        torch.save(snapshot(model, opt, steps, config=asdict(c)), path)
        schedule(opt, steps, 305176, wd)
        target = train_step(model, opt, data, steps)
        coefficients = {k:v.detach().clone() for k,v in model.chord.layers.named_parameters()}
        del opt, model, active
        gc.collect()
        state = torch.load(path, map_location='cpu', weights_only=False)
        restored = make(ConnectionConfig.from_checkpoint(state['config']), True)
        opt, wd = optimizer_for(restored)
        restored.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        restore_rng(state['rng'])
        schedule(opt, state['step'], 305176, wd)
        replay = train_step(restored, opt, data, state['step'])
        assert abs(replay-target) < 2e-5, (replay, target)
        for name, value in restored.chord.layers.named_parameters():
            torch.testing.assert_close(value, coefficients[name], rtol=1e-4, atol=2e-5)
    return dict(steps=steps, seconds_per_step=statistics.median(times[5:]), final_train_ce=loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token-vnorm', action='store_true')
    token = parser.parse_args().token_vnorm
    configure_attention('fa3')
    seed_all(42)
    if token:
        from research.connections.native.chord.token_control_checks import check_token_controls
        check_token_controls()
    else:
        kernel_checks()
    c = ConnectionConfig(architecture='native', variant='baseline', streams=1,
        native_chord_write='token_vnorm' if token else 'three_scalar', residual_dynamics='ngpt_lookahead',
        ngpt_velocity_mode='lookahead_split_learned_beta', chord_velocity_geometry='project_only',
        velocity_init='token', fuse_chord_read_norm=True, ngpt_force_rms_floor=1e-6)
    control = replace(c, native_chord_write='three_scalar' if token else 'tied')
    data = PackedData('../data')
    model, tied = make(c, False), make(control, False)
    assert set(model.native.value_embeds) == {'1','3','5','7'}
    extra = 2*c.width*(6*(2*c.depth-1) + 3) if token else 30
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in tied.parameters()) == extra
    ids = data.batch('train', 0, 1)[0][:, :32]
    paired_and_causal(model, tied, ids)
    del model, tied
    gc.collect()
    result = {}
    with tempfile.TemporaryDirectory(dir='..') as tmp:
        for cfg in (control, c):
            result[cfg.native_chord_write] = speed_and_resume(cfg, data, Path(tmp), check_resume=(cfg == c), steps=128 if token else 16)
            print(cfg.native_chord_write, result[cfg.native_chord_write], flush=True)
    result.update(gpu=torch.cuda.get_device_name(), kernel_vjp='passed', restart='passed',
                  initialization_parity='passed', causal_and_kv='passed', extra_parameters=extra)
    filename = 'native-chord-token-vnorm-sm90.json' if token else 'native-chord-sm90.json'
    json_write(Path('results') / filename, result)
    print('NATIVE_CHORD_PREFLIGHT_PASS', result, flush=True)


if __name__ == '__main__':
    main()
