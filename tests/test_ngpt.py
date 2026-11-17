"""Pure nGPT residual: independent oracle plus shared end-to-end GPU checks."""
from dataclasses import asdict, replace
import math

import pytest
import torch

from nanochat.connection_model import ConnectionConfig
from nanochat.connection_cost import model_cost
from nanochat.momentum import NormalizedResidual, inverse_softplus, normalize_residual
# Reuse the same finite/causal/compile/checkpoint standards as the hybrid.
from test_momentum import (
    gpu, inputs, make,
    test_backbone_weights_and_optimizer_membership,
    test_compiled_checkpoint_forward_backward,
    test_cce_gradient_and_velocity_learning,
    test_causal_batch_and_cached_chunks,
    test_exact_optimizer_checkpoint_next_update,
)


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32,
                            vocab_size=256, streams=1, residual_dynamics='ngpt')


def test_single_state_parameters_and_config(setting):
    model = make(setting, perturb=False)
    assert not hasattr(model, 'velocity_embedding')
    assert not any('momentum' in n or 'velocity' in n for n, _ in model.named_parameters())
    residual = [p for n, p in model.named_parameters() if '_residual.' in n]
    assert len(residual) == 2*setting.depth and all(p.numel() == 1 for p in residual)
    assert model_cost(setting)['additional_parameters'] == 2*setting.depth
    assert setting.velocity_init == 'zero'
    assert ConnectionConfig.from_checkpoint(asdict(setting)) == setting
    with pytest.raises(ValueError, match='momentum stream'):
        replace(setting, velocity_init='token')
    x, _ = inputs()
    h = normalize_residual(model.wte(x))
    out, layers = model.backbone(h, return_layers=True)
    torch.testing.assert_close(out, h, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(layers.square().mean(-1), torch.ones_like(layers[..., 0]), rtol=0, atol=5e-7)


@pytest.mark.parametrize('alpha', [.01, .05, .3, .8])
def test_official_scalar_lerp_and_gradients(setting, alpha):
    update = NormalizedResidual(setting).cuda().double()
    with torch.no_grad():
        update.raw.fill_(inverse_softplus(alpha))
    h, f = [torch.randn(2, 128, device='cuda', dtype=torch.float64, requires_grad=True) for _ in range(2)]
    h = normalize_residual(h)
    actual = update(h, f)
    # Independent NVIDIA unit-radius formula, rescaled to sqrt(d).
    a, b = [x/x.norm(dim=-1, keepdim=True) for x in (h, f)]
    z = a + update.alpha()*(b-a)
    expected = z/z.norm(dim=-1, keepdim=True)*math.sqrt(128)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    probe = torch.randn_like(actual)
    ga = torch.autograd.grad((actual*probe).sum(), (h, f, update.raw), retain_graph=True)
    gb = torch.autograd.grad((expected*probe).sum(), (h, f, update.raw))
    # h is an already-normalized state; compare along its tangent directions.
    tangent = lambda g: g-(g*h).mean(-1, keepdim=True)*h
    torch.testing.assert_close(tangent(ga[0]), tangent(gb[0]), rtol=1e-11, atol=1e-11)
    for left, right in zip(ga[1:], gb[1:]):
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('zero', [False, True])
def test_first_second_derivatives_and_zero_branch(zero):
    update = NormalizedResidual(ConnectionConfig(residual_dynamics='ngpt')).cuda().double()
    h = torch.randn(1, 4, device='cuda', dtype=torch.float64, requires_grad=True)
    f = (torch.zeros_like(h) if zero else torch.randn_like(h)).requires_grad_()
    raw = torch.tensor([inverse_softplus(.05)], device='cuda', dtype=torch.float64, requires_grad=True)
    def fn(h, f, raw):
        return torch.func.functional_call(update, {'raw': raw}, (normalize_residual(h), f))
    assert torch.autograd.gradcheck(fn, (h, f, raw), eps=1e-6, atol=2e-6, rtol=1e-4)
    assert torch.autograd.gradgradcheck(fn, (h, f, raw), eps=1e-6, atol=4e-6, rtol=1e-4)


def test_32_layer_finite_and_alpha_learns(setting):
    model = make(replace(setting, depth=32))
    x, y = inputs()
    loss = model(x, y, dense=True)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert all(m.raw.grad.abs().sum() > 0 for b in model.blocks for m in (b.attn_residual, b.mlp_residual))
    h = normalize_residual(model.wte(x))
    _, layers = model.backbone(h, return_layers=True)
    torch.testing.assert_close(layers.square().mean(-1), torch.ones_like(layers[..., 0]), rtol=0, atol=8e-7)
