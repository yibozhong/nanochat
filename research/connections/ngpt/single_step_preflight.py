"""Same-allocation GPU checks for vanilla h-only, three-control Chord."""
import argparse
import gc
import math
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from nanochat.chord_single_step import reference_read, reference_write
from nanochat.chord_single_step_triton import TokenRead, TokenWrite
from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
from nanochat.engine import KVCache
from nanochat.packed_training import PackedData, evaluate, json_write, restore_rng, seed_all, snapshot, train_step
from research.connections.native.chord.preflight import assert_exact_state, connector_group
from research.connections.native.chord.token_control_checks import check_token_controls, check_vjp


def kernel_checks(c):
    for terminal in (False, True):
        def rand(*shape):
            return torch.randn(shape, device='cuda')
        h, v, f = rand(2, 71, c.width).bfloat16(), rand(2, 71, c.width).bfloat16(), rand(2, 71, c.width).bfloat16()
        v[:, 0] = 0
        h[:, :3] *= .02  # Distinguishes RMSNorm(h) from a raw-h projection.
        w = rand(2 if terminal else 3, c.width) * (.1 / c.width**.5)
        args = (h, v, w, rand(1)*.1, rand(1)-2, None if terminal else rand(1))
        leaves = [x.requires_grad_() for x in args if x is not None]
        q, controls = TokenRead.apply(*args)
        assert not torch.count_nonzero(torch.autograd.grad(controls.sum(), v, retain_graph=True)[0])
        check_vjp(reference_read(*args), (q, controls), leaves, bounded_read=True)
        f[:, :2] = 0
        write_args = (h, v, f.requires_grad_(), controls.detach().requires_grad_())
        fast = TokenWrite.apply(*write_args, c.ngpt_force_rms_floor)
        check_vjp(reference_write(*write_args, c.ngpt_force_rms_floor), fast, write_args)
        pos, vel = (x.float() for x in fast)
        assert (pos.square().mean(-1)-1).abs().max().item() < .004
        assert (pos*vel).mean(-1).abs().max().item() < .004


def make(c, compiled):
    seed_all(42)
    return build_connection_model(c).prepare(compile_blocks=compiled)


@torch.no_grad()
def initialization_and_cache(c, baseline, data):
    m, b = make(c, False).eval(), make(baseline, False).eval()
    small_dynamic = c.chord_six_controls == 'small_dynamic'
    for name, p in m.named_parameters():
        if (small_dynamic and not name.endswith('.token_weight')) or '_residual.' not in name:
            assert torch.equal(p, b.get_parameter(name)), name
    ids = data.batch('dev', 0, 1)[0][:, :32].contiguous()
    hidden, control = m.hidden(ids), b.hidden(ids)
    assert (hidden.float()-control.float()).norm() < .003 * control.float().norm()
    for block in m.blocks:
        for site in (block.attn_residual, block.mlp_residual):
            std, mean = site.eval_control_moments if small_dynamic else site.eval_moments
            expected = [0, .05, .05, .5, .5, .5] if small_dynamic else [0, .05, .5]
            torch.testing.assert_close(mean, mean.new_tensor(expected))
            assert torch.count_nonzero(std)==0
            weight = site.token_weight if small_dynamic else site.weight
            weight.normal_(std=(20 if small_dynamic else .2)/c.width**.5)
    hidden = m.hidden(ids)
    changed = ids.clone()
    changed[:, 16:] = ids[:, 16:].roll(1, 1)
    torch.testing.assert_close(hidden[:, :16], m.hidden(changed)[:, :16], atol=0, rtol=0)
    cache = KVCache(1, c.heads, 32, c.width//c.heads, c.depth, 'cuda', torch.bfloat16)
    prefix = m.hidden(ids[:, :-1].contiguous(), cache)
    last = m.hidden(ids[:, -1:], cache)
    assert cache.get_pos() == ids.size(1)
    torch.testing.assert_close(torch.cat((prefix, last), 1), hidden, atol=.035, rtol=.02)


def benchmark(c, data, verify_resume):
    model = make(c, True)
    opt, wd = optimizer_for(model)
    group = connector_group(opt, (p for b in model.blocks for site in (b.attn_residual, b.mlp_residual)
                                  for p in site.parameters()))
    expected_lr = .008 * (32768 / 2**19)**.5 * (c.width / 768)**-.5
    assert group['kind'] == 'adamw' and math.isclose(group['initial_lr'], expected_lr)
    assert group['betas'] == (.8, .96) and group['weight_decay'] == 0
    times = []
    steps = 64
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        schedule(opt, step, 305176, wd)
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            loss = train_step(model, opt, data, step)
        except FloatingPointError as error:
            bad = [name for name, p in model.named_parameters()
                   if p.grad is not None and not torch.isfinite(p.grad).all()]
            raise FloatingPointError(f'{c.ngpt_velocity_mode}/{c.chord_six_controls} at step {step}: '
                                     f'nonfinite gradients in {bad}') from error
        torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
    result = dict(steps=steps, median_step_s=statistics.median(times[16:]),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), last_train_ce=loss,
                  control_peak_lr=group['initial_lr'])
    result['dev_ce'], result['dev_tokens'] = evaluate(model, data, max_rows=64)
    assert math.isfinite(result['dev_ce'])
    print('64_STEP_MEASUREMENT', c.ngpt_velocity_mode, c.chord_six_controls, result, flush=True)
    if verify_resume:
        for block in model.blocks:
            for site in (block.attn_residual, block.mlp_residual):
                if c.chord_update_based:
                    for param in site.parameters():
                        assert param.grad is not None and torch.isfinite(param.grad).all()
                elif c.ngpt_velocity_mode == 'lookahead_minimal':
                    assert site.mu.grad is not None and site.mu.grad.abs().item() > 0
                    if site.raw_beta is not None:
                        assert site.raw_beta.grad is not None and site.raw_beta.grad.abs().item() > 0
                else:
                    weight = site.token_weight if c.chord_six_controls == 'small_dynamic' else site.weight
                    assert (weight.grad.flatten(1).norm(dim=1)>0).all()
        with TemporaryDirectory(dir='..') as tmp:
            path = Path(tmp)/'resume.pt'
            torch.save(snapshot(model, opt, steps, config=asdict(c)), path)
            state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['step'] == steps
        restored = make(ConnectionConfig.from_checkpoint(state['config']), True)
        other, wd2 = optimizer_for(restored)
        restored.load_state_dict(state['model'])
        other.load_state_dict(state['optimizer'])
        assert_exact_state(model.state_dict(), restored.state_dict())
        assert_exact_state(opt.state_dict(), other.state_dict())
        rng = state.pop('rng')
        del state
        restore_rng(rng)
        schedule(opt, steps, 305176, wd)
        expected = train_step(model, opt, data, steps)
        restore_rng(rng)
        schedule(other, steps, 305176, wd2)
        def paired_gradient(optimizer, args, kwargs):
            for (name, p), (restored_name, q) in zip(model.named_parameters(), restored.named_parameters()):
                assert name == restored_name and (p.grad is None) == (q.grad is None)
                if p.grad is not None:
                    q.grad.copy_(p.grad)
        hook = other.register_step_pre_hook(paired_gradient)
        try:
            actual = train_step(restored, other, data, steps)
        finally:
            hook.remove()
        assert abs(expected-actual)<2e-5
        assert_exact_state(model.state_dict(), restored.state_dict())
        assert_exact_state(opt.state_dict(), other.state_dict())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--small-dynamic', action='store_true')
    small_dynamic = parser.parse_args().small_dynamic
    configure_attention('fa2')
    torch.backends.cuda.matmul.allow_tf32 = False
    c = ConnectionConfig(streams=1, residual_dynamics='ngpt_lookahead', ngpt_velocity_mode='lookahead_token_single',
                         chord_velocity_geometry='project_only', velocity_init='token',
                         fuse_chord_read_norm=True, ngpt_force_rms_floor=1e-6)
    baseline = replace(c, ngpt_velocity_mode='lookahead_split_learned_beta')
    if small_dynamic:
        c = replace(baseline, chord_six_controls='small_dynamic')
        baseline = replace(baseline, chord_six_controls='static')
    data = PackedData('../data')
    seed_all(42)
    if small_dynamic:
        check_token_controls(position_only=True, small_dynamic=True)
    else:
        kernel_checks(c)
    initialization_and_cache(c, baseline, data)
    result = dict(gpu=torch.cuda.get_device_name(), kernel_vjp=True, cache_and_causal=True)
    name = 'small_dynamic' if small_dynamic else 'token_single'
    for key, cfg in [(name, c), ('static_chord', baseline)]:
        gc.collect()
        torch.cuda.empty_cache()
        result[key] = benchmark(cfg, data, key==name)
        print(key, result[key], flush=True)
    result['step_time_ratio'] = result[name]['median_step_s']/result['static_chord']['median_step_s']
    assert result[name]['control_peak_lr'] == result['static_chord']['control_peak_lr']
    report = 'vanilla-small-dynamic-sm90.json' if small_dynamic else 'vanilla-single-sm90.json'
    json_write(Path('results')/report, result)
    print('VANILLA_SINGLE_PREFLIGHT_PASS', result, flush=True)


if __name__ == '__main__':
    main()
