"""CUDA gates for the spherical branch lookahead experiment."""
from dataclasses import replace

import pytest
import torch

from nanochat.connection_model import ConnectionConfig, optimizer_for
from nanochat.lookahead import LookaheadResidual
from nanochat.momentum import normalize_residual, tangent_project
from nanochat.packed_training import seed_all
from test_momentum import (
    gpu, inputs, make, test_causal_batch_and_cached_chunks,
    test_cce_gradient_and_velocity_learning, test_compiled_checkpoint_forward_backward,
    test_exact_optimizer_checkpoint_next_update,
)
from test_nonzero_residual import test_32_layer_production_initialization


@pytest.fixture
def setting():
    # Exercise cache, gradients and recovery with the velocity read already active.
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_lookahead', lookahead_mu_init=.3,
                            residual_projection_init='normal', ngpt_force_rms_floor=1e-6)


def test_zero_lookahead_matches_baseline_and_learns(setting):
    config = replace(setting, lookahead_mu_init=0.)
    seed_all(42)
    baseline = 'vanilla' if setting.residual_mode == 'lookahead_only' else 'ngpt'
    reference = make(replace(config, residual_dynamics=baseline, velocity_init='zero', ngpt_velocity_mode=None))
    seed_all(42)
    model = make(config)
    params = dict(model.named_parameters())
    for name, p in reference.named_parameters():
        assert torch.equal(p, params[name]), name
    x, y = inputs()
    assert torch.equal(reference(x), model(x))
    ref_loss, loss = reference(x, y, dense=True), model(x, y, dense=True)
    assert torch.equal(ref_loss, loss)
    ref_loss.backward()
    loss.backward()
    for name, p in reference.named_parameters():
        diff = (p.grad.float() - params[name].grad.float()).norm()
        assert diff <= .01 * p.grad.float().norm() + 1e-10, name
    sites = [s for b in model.blocks for s in (b.attn_residual, b.mlp_residual)]
    assert all(torch.isfinite(s.mu.grad).all() and s.mu.grad.abs().sum() > 0 for s in sites)
    tables = (model.velocity_embedding.weight, model.velocity_position_embedding.weight)
    assert all(torch.count_nonzero(p.grad) == 0 for p in tables)
    opt, _ = optimizer_for(model)
    opt.step()
    assert all(s.mu.item() != 0 for s in sites)
    opt.zero_grad(set_to_none=True)
    model(x, y, dense=True).backward()
    assert all(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in tables)


def test_geometry_and_derivatives(setting):
    site = LookaheadResidual(replace(setting, ngpt_alpha_init=.5)).cuda().double()
    # A quarter-turn target makes the actual nGPT step a 45-degree turn.
    h = torch.tensor([[2., 0., 0., 0.]], device='cuda', dtype=torch.double)
    force = h.roll(1, -1)
    v = torch.tensor([[0., .8, -.3, .4]], device='cuda', dtype=torch.double)
    new_h, new_v = site(h, v, force)
    cos, sin = new_h[0, 0]/2, new_h[0, 1]/2
    transported = torch.stack((-sin*v[0, 1], cos*v[0, 1], v[0, 2], v[0, 3]))[None, :]
    observed = tangent_project(new_h, new_h-h)
    torch.testing.assert_close(new_v, site.decay*transported + (1-site.decay)*observed)
    torch.testing.assert_close((new_h*new_v).sum(), new_h.new_zeros(()))
    torch.testing.assert_close(site.query(h, v), normalize_residual(h + site.mu*v))

    def recurrence(raw_h, raw_v, branch):
        state = normalize_residual(raw_h)
        memory = tangent_project(state, raw_v)
        return (site.query(state, memory), *site(state, memory, branch))

    args = tuple(t.clone().requires_grad_() for t in (h, v, force))
    assert torch.autograd.gradcheck(recurrence, args)
