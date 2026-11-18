"""GPU validation for affine velocity normalization in the spherical HBFormer."""
from dataclasses import asdict, replace

import pytest
import torch

from nanochat.connection_model import ConnectionConfig, optimizer_for
from nanochat.momentum import NormalizedHeavyBall
from test_momentum import (
    gpu, inputs, make, test_compiled_checkpoint_forward_backward, test_cce_gradient_and_velocity_learning,
    test_causal_batch_and_cached_chunks, test_exact_optimizer_checkpoint_next_update,
)
from test_nonzero_residual import test_32_layer_production_initialization


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_hbformer', ngpt_velocity_mode='transport_layernorm',
                            ngpt_force_rms_floor=1e-6)


def test_learned_speed_and_transport(setting):
    update = NormalizedHeavyBall(setting).cuda().double()
    h, v, f = torch.randn(3, 4, setting.width, dtype=torch.double, device='cuda')
    h = h / h.square().mean(-1, keepdim=True).sqrt()
    # Unequal gains expose LN/projection order. Large gains expose an extra RMS.
    with torch.no_grad():
        update.velocity_norm.weight.copy_(torch.linspace(1., 4., setting.width, device=h.device))
    beta, gamma, eta = update.coefficients()
    assert eta == 1 and not eta.requires_grad
    mixed = beta*v + gamma*f/f.square().mean(-1, keepdim=True).sqrt()
    centered = mixed - mixed.mean(-1, keepdim=True)
    affine = centered * (centered.square().mean(-1, keepdim=True) + 1e-5).rsqrt() * update.velocity_norm.weight
    speed = affine - (affine*h).mean(-1, keepdim=True)*h
    rms2 = speed.square().mean(-1, keepdim=True)
    assert (rms2 > 1).all()
    new_h, carry = update(h, v, f)
    # Closed form when the transported vector is the retraction direction.
    torch.testing.assert_close(new_h, (h+speed)/(1+rms2).sqrt(), atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(carry, (speed-rms2*h)/(1+rms2).sqrt(), atol=1e-11, rtol=1e-11)
    torch.testing.assert_close((new_h*carry).mean(-1), torch.zeros_like(h[:, 0]), atol=1e-11, rtol=0)
    with torch.no_grad():
        update.velocity_norm.weight.zero_()
    identity, stopped = update(h, v, f)
    torch.testing.assert_close(identity, h)
    assert stopped.count_nonzero() == 0


def test_gradients_through_gain_projection_and_transport(setting):
    update = NormalizedHeavyBall(replace(setting, width=4, heads=1)).cuda().double()
    def recurrence(x, v, f, raw, gain):
        h = x / x.square().mean(-1, keepdim=True).sqrt()
        params = {'raw':raw, 'velocity_norm.weight':gain}
        return torch.func.functional_call(update, params, (h, v, f))
    args = tuple(torch.randn(2,4,device='cuda',dtype=torch.double,requires_grad=True) for _ in range(3))
    args += (update.raw.detach().clone().requires_grad_(),
             update.velocity_norm.weight.detach().clone().requires_grad_())
    assert torch.autograd.gradcheck(recurrence, args)


def test_gain_initialization_optimizer_and_metrics(setting):
    assert setting.ngpt_weight_norm is False
    assert ConnectionConfig.from_checkpoint(asdict(setting)) == setting
    model = make(setting)
    modules = [m for b in model.blocks for m in (b.attn_residual, b.mlp_residual)]
    gains = [m.velocity_norm.weight for m in modules]
    assert len({id(g) for g in gains}) == 2*setting.depth
    assert all(m.velocity_norm.bias is None and m.velocity_norm.eps == 1e-5 for m in modules)
    opt, _ = optimizer_for(model)
    for gain in gains:
        torch.testing.assert_close(gain, torch.ones_like(gain))
        group, = [group for group in opt.param_groups if any(p is gain for p in group['params'])]
        assert group['kind'] == 'adamw' and group['weight_decay'] == 0.
    x, y = inputs()
    model(x,y).backward()
    assert all(g.grad is not None and torch.isfinite(g.grad).all() and g.grad.norm() > 0 for g in gains)
    opt.step()
    assert all((g != 1).any() for g in gains)
    assert model.residual_metrics()['residual/sublayer_0/velocity_gain_std'] > 0
