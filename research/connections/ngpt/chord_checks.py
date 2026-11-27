"""GPU checks used by the Next Chord preflight and its scalar timing reference."""
from dataclasses import asdict
import math
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
import time

import torch

from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
from nanochat.packed_training import evaluate, restore_rng, seed_all, snapshot, train_step


def check_read_vjp(expected, actual, inputs):
    expected, actual = tuple(x for x in expected if x is not None), tuple(x for x in actual if x is not None)
    probes = [torch.randn_like(x) for x in actual]
    for x, y in zip(actual, expected):
        if x.dtype == torch.bfloat16:
            assert (x.float()-y.float()).norm() < .003 * y.float().norm()
        else:
            torch.testing.assert_close(x, y, atol=3e-5, rtol=2e-5)
    lhs = torch.autograd.grad(actual, inputs, probes)
    rhs = torch.autograd.grad(expected, inputs, probes)
    for x, y in zip(lhs, rhs):
        diff = (x.float()-y.float()).norm() / y.float().norm().clamp_min(1e-7)
        assert torch.isfinite(x).all() and diff < .015, (x.shape, diff.item())


def make(c, compiled):
    seed_all(42)
    return build_connection_model(c).prepare(compile_blocks=compiled)


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


def connector_group(optimizer, parameters):
    ids = {id(p) for p in parameters}
    group, = [g for g in optimizer.param_groups if any(id(p) in ids for p in g['params'])]
    assert {id(p) for p in group['params']} == ids
    return {k: v for k, v in group.items() if k != 'params'}


def benchmark(c, data, verify_resume, schedule_steps):
    model = make(c, True)
    opt, wd = optimizer_for(model)
    if c.velocity_init == 'projected_h0':
        matrix, = (g for g in opt.param_groups if any(p is model.velocity_projection.weight for p in g['params']))
        assert matrix['kind'] == 'muon' and math.isclose(matrix['initial_lr'], .005)
        assert any(p is model.blocks[0].attn.c_q.weight for p in matrix['params'])
    group = connector_group(opt, (p for b in model.blocks for site in (b.attn_residual, b.mlp_residual)
                                  for p in site.parameters()))
    expected_lr = .008 * (32768 / 2**19)**.5 * (c.width / 768)**-.5
    assert group['kind'] == 'adamw' and math.isclose(group['initial_lr'], expected_lr)
    assert group['betas'] == (.8, .96) and group['weight_decay'] == 0
    times = []
    steps = 64
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        schedule(opt, step, schedule_steps, wd)
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
                if c.chord_zero_mu:
                    # Check every actual model site after training, including terminal FFN.
                    assert site.mu.grad.count_nonzero() == site.token_weight.grad[0].count_nonzero() == 0
                    assert site.mix_scales.grad[0].item() == 0
                    std, mean = site.eval_control_moments
                    assert std[0].item() == mean[0].item() == 0
                for param in site.parameters():
                    assert param.grad is not None and torch.isfinite(param.grad).all()
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
        schedule(opt, steps, schedule_steps, wd)
        expected = train_step(model, opt, data, steps)
        restore_rng(rng)
        schedule(other, steps, schedule_steps, wd2)
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
