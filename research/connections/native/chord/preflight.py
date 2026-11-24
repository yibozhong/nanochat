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
    if model.config.native_chord_write in ('token_bounded', 'token_h'):
        for layer in model.chord.layers:
            for site in (layer.attention, layer.ffn):
                std, mean = site.eval_control_moments
                torch.testing.assert_close(mean, mean.new_tensor([0, .05, .05, .5, .5, .5]))
                assert torch.count_nonzero(std) == 0
    if model.chord.layers[0].token_controls:
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


def assert_exact_state(expected, actual):
    from torch.utils._pytree import tree_flatten
    expected_values, expected_structure = tree_flatten(expected)
    actual_values, actual_structure = tree_flatten(actual)
    assert expected_structure == actual_structure
    for left, right in zip(expected_values, actual_values):
        if isinstance(left, torch.Tensor):
            assert left.dtype == right.dtype and left.device == right.device and torch.equal(left, right)
        else:
            assert left == right


def speed_and_resume(c, data, folder, *, check_resume, steps):
    model = make(c, True)
    opt, wd = optimizer_for(model)
    if c.native_chord_control_lr is not None:
        controls = {id(p) for p in model.chord.layers.parameters()}
        group, = [g for g in opt.param_groups if any(id(p) in controls for p in g['params'])]
        assert {id(p) for p in group['params']} == controls and group['kind'] == 'adamw'
        assert group['initial_lr'] == group['lr'] == c.native_chord_control_lr
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
        if model.chord.layers[0].token_controls:
            for layer in model.chord.layers:
                for site in (layer.attention, layer.ffn):
                    grad = site.token_weight.grad
                    assert grad is not None and (grad.float().norm(dim=-1) > 0).all()
        path = folder/'local.pt'
        torch.save(snapshot(model, opt, steps, config=asdict(c)), path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        restored = make(ConnectionConfig.from_checkpoint(state['config']), True)
        other_opt, other_wd = optimizer_for(restored)
        restored.load_state_dict(state['model'])
        other_opt.load_state_dict(state['optimizer'])
        assert state['step'] == steps and restored.config == c and other_wd == wd
        assert_exact_state(model.state_dict(), restored.state_dict())
        assert_exact_state(opt.state_dict(), other_opt.state_dict())
        rng = state.pop('rng')
        del state
        # Both models run the real NTP forward and backward at the resumed data
        # row. Isolate optimizer restoration from independent BF16 reductions:
        # the native resume oracle applies precisely the same already-finite gradients.
        def replay_gradients(optimizer, args, kwargs):
            for (name, live), (other_name, other) in zip(model.named_parameters(), restored.named_parameters()):
                assert name == other_name and (live.grad is None) == (other.grad is None)
                if live.grad is not None:
                    other.grad.copy_(live.grad)

        restore_rng(rng)
        schedule(opt, steps, 305176, wd)
        target = train_step(model, opt, data, steps)
        restore_rng(rng)
        schedule(other_opt, steps, 305176, wd)
        hook = other_opt.register_step_pre_hook(replay_gradients)
        try:
            replay = train_step(restored, other_opt, data, steps)
        finally:
            hook.remove()
        assert abs(replay-target) < 2e-5, (replay, target)
        assert_exact_state(model.state_dict(), restored.state_dict())
        assert_exact_state(opt.state_dict(), other_opt.state_dict())
        print(f'Resume verified at {steps}: exact loaded states and fixed-gradient update; '
              f'next batch CE {target:.8f} / {replay:.8f}', flush=True)
    return dict(steps=steps, seconds_per_step=statistics.median(times[5:]), final_train_ce=loss)


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--token-vnorm', dest='mode', action='store_const', const='token_vnorm')
    modes.add_argument('--token-bounded', dest='mode', action='store_const', const='token_bounded')
    modes.add_argument('--token-h', dest='mode', action='store_const', const='token_h')
    parser.add_argument('--native-chord-control-lr', type=float)
    args = parser.parse_args()
    mode = args.mode or 'three_scalar'
    token = mode != 'three_scalar'
    configure_attention('fa3')
    seed_all(42)
    if token:
        from research.connections.native.chord.token_control_checks import check_token_controls
        check_token_controls(bounded=(mode == 'token_bounded'), position_only=(mode == 'token_h'))
    else:
        kernel_checks()
    c = ConnectionConfig(architecture='native', variant='baseline', streams=1,
        native_chord_write=mode, native_chord_control_lr=args.native_chord_control_lr, residual_dynamics='ngpt_lookahead',
        ngpt_velocity_mode='lookahead_split_learned_beta', chord_velocity_geometry='project_only',
        velocity_init='token', fuse_chord_read_norm=True, ngpt_force_rms_floor=1e-6)
    comparison = {'three_scalar': 'tied', 'token_vnorm': 'three_scalar', 'token_bounded': 'token_vnorm',
                  'token_h': 'token_vnorm'}
    control = replace(c, native_chord_write=comparison[mode])
    data = PackedData('../data')
    model, tied = make(c, False), make(control, False)
    assert set(model.native.value_embeds) == {'1','3','5','7'}
    one_stream = c.width * (6*(2*c.depth-1) + 3)
    extra = {'three_scalar': 30, 'token_vnorm': 2*one_stream, 'token_bounded': 0, 'token_h': -one_stream}[mode]
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
    result.update(gpu=torch.cuda.get_device_name(), kernel_vjp='passed', restart='exact state and fixed-gradient update, paired next-batch CE',
                  initialization_parity='passed', causal_and_kv='passed', extra_parameters=extra)
    filename = f"native-chord-{mode.replace('_', '-')}-sm90.json" if token else 'native-chord-sm90.json'
    json_write(Path('results') / filename, result)
    print('NATIVE_CHORD_PREFLIGHT_PASS', result, flush=True)


if __name__ == '__main__':
    main()
