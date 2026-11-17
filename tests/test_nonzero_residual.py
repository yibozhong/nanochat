"""Production nonzero-init residual configurations, on a Condor GPU only."""
from dataclasses import asdict, replace
import math

import pytest
import torch

from nanochat.connection_model import ConnectionConfig
from nanochat.momentum import NormalizedResidual, normalize_residual, ngpt_momentum_step
from scripts.connection_momentum_debug import residual_audit
from test_momentum import (
    gpu, inputs, make,
    test_backbone_weights_and_optimizer_membership,
    test_compiled_checkpoint_forward_backward,
    test_cce_gradient_and_velocity_learning,
    test_causal_batch_and_cached_chunks,
    test_exact_optimizer_checkpoint_next_update,
)

MODES = ('vanilla', 'ngpt', 'ngpt_heavy_ball')


@pytest.fixture(params=MODES)
def setting(request):
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32,
        vocab_size=256, streams=1, residual_dynamics=request.param,
        residual_projection_init='normal', residual_projection_init_std=.02,
        ngpt_force_rms_floor=1e-6)


def test_initialization_and_first_step_gradients(setting):
    model = make(setting, perturb=False)
    sigma = .02/math.sqrt(2*setting.depth)
    for b in model.blocks:
        for p in (b.attn.c_proj.weight, b.mlp.c_proj.weight):
            assert .97 < p.float().std().item()/sigma < 1.03
            assert p.abs().sum() > 0
    x, y = inputs()
    loss = model(x, y, dense=True)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        # With nonzero output projections, inner branch weights learn on step 1.
        assert p.grad.abs().sum() > 0, name
    audit = residual_audit(model, x, require_unit_target=True)
    if setting.residual_mode != 'vanilla':
        assert all(row['force_floor_fraction'] == 0 for row in audit)
    assert ConnectionConfig.from_checkpoint(asdict(setting)) == setting
    assert (getattr(model, 'velocity_embedding', None) is not None) == (setting.residual_mode == 'ngpt_heavy_ball')


@pytest.mark.parametrize('scale', [1e-4, .01, 1.])
def test_exact_ngpt_direction_even_for_small_nonzero_branches(scale):
    config = ConnectionConfig(residual_dynamics='ngpt', ngpt_force_rms_floor=1e-6,
                              residual_projection_init='normal')
    update = NormalizedResidual(config).cuda().double()
    h = normalize_residual(torch.randn(3, 128, device='cuda', dtype=torch.float64))
    f = scale*torch.randn_like(h)
    a, b = [x/x.norm(dim=-1, keepdim=True) for x in (h, f)]
    z = a + update.alpha()*(b-a)
    expected = z/z.norm(dim=-1, keepdim=True)*math.sqrt(128)
    torch.testing.assert_close(update(h, f), expected, atol=1e-12, rtol=1e-12)
    actual, _ = ngpt_momentum_step(h, torch.randn_like(h), f, h.new_tensor(0.),
                                   h.new_tensor(.5), 2*update.alpha(), 1e-6)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_nonzero_ngpt_first_and_second_derivatives():
    config = ConnectionConfig(residual_dynamics='ngpt', ngpt_force_rms_floor=1e-6,
                              residual_projection_init='normal')
    update = NormalizedResidual(config).cuda().double()
    h = torch.randn(1, 4, device='cuda', dtype=torch.float64, requires_grad=True)
    f = (.01*torch.randn_like(h)).requires_grad_()
    raw = update.raw.detach().clone().requires_grad_()
    def fn(h, f, raw):
        return torch.func.functional_call(update, {'raw': raw}, (normalize_residual(h), f))
    assert torch.autograd.gradcheck(fn, (h, f, raw), eps=1e-7, atol=2e-6, rtol=1e-4)
    assert torch.autograd.gradgradcheck(fn, (h, f, raw), eps=1e-7, atol=4e-6, rtol=1e-4)


def test_32_layer_production_initialization(setting):
    model = make(replace(setting, depth=32), perturb=False)
    x, y = inputs()
    loss = model(x, y, dense=True)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert len(residual_audit(model, x, require_unit_target=True)) == 64


def test_legacy_initialization_defaults():
    old = asdict(ConnectionConfig())
    old.pop('residual_projection_init'); old.pop('residual_projection_init_std')
    assert ConnectionConfig.from_checkpoint(old).residual_projection_init == 'zero'
    for fields in ({'residual_projection_init':'bad'}, {'residual_projection_init_std':0},
                   {'residual_projection_init_std':float('nan')},
                   {'residual_projection_init':'normal', 'architecture':'native'}):
        with pytest.raises(ValueError):
            ConnectionConfig(**fields)
