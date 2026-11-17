"""GPU checks for the unit-step, weight-normalized HBFormer combination."""
import math

import pytest
import torch

from nanochat.connection_model import ConnectionConfig, optimizer_for
from test_momentum import (
    gpu, inputs, make, test_compiled_checkpoint_forward_backward,
    test_cce_gradient_and_velocity_learning, test_causal_batch_and_cached_chunks,
    test_exact_optimizer_checkpoint_next_update,
)
from test_nonzero_residual import test_32_layer_production_initialization


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_hbformer', ngpt_force_rms_floor=1e-6)


def test_original_velocity_and_unit_step(setting):
    model = make(setting)
    x, _ = inputs()
    start = 4
    h, v = model.initial_state(model.wte(x), x, start)
    token, pos = model.velocity_embedding.weight, model.velocity_position_embedding.weight
    expected = token[x].float() + pos[start:start+x.shape[1]].float()
    torch.testing.assert_close(v, expected, atol=0, rtol=0)
    assert all(.018 < p.float().std() < .022 for p in (token, pos))
    update = model.blocks[0].attn_residual
    beta, gamma, eta = update.coefficients()
    assert update.raw.numel() == 2 and not eta.requires_grad and eta.item() == 1
    assert beta.item() == .5 and abs(gamma.item()-math.log(2.)) < 1e-7
    force = torch.randn_like(h)
    target = torch.nn.functional.normalize(force, dim=-1)*setting.width**.5
    u = beta*v + gamma*target
    u = u - (u*h).mean(-1, keepdim=True)*h
    speed = u / u.square().mean(-1, keepdim=True).sqrt().clamp_min(.1)
    hidden, velocity = update(h, v, force)
    ref = torch.nn.functional.normalize(h+speed, dim=-1)*setting.width**.5
    torch.testing.assert_close(hidden, ref, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close((hidden*velocity).mean(-1), torch.zeros_like(h[...,0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(velocity.norm(dim=-1), speed.norm(dim=-1))


def test_nvidia_weight_axes_and_optimizer_step(setting):
    model = make(setting)
    opt, _ = optimizer_for(model)
    x, y = inputs()
    for stage in ('init', 'after_optimizer_step'):
        for name, p in model.named_parameters():
            if p.ndim != 2 or name.startswith('velocity_'):
                continue
            dim = 0 if name.endswith('c_proj.weight') else 1
            lengths = p.float().norm(dim=dim)
            tol = .002 if p.dtype == torch.bfloat16 else 2e-6
            torch.testing.assert_close(lengths, torch.ones_like(lengths), atol=tol, rtol=0)
        if stage == 'init':
            # Force a radial error to ensure the optimizer hook actually projects.
            with torch.no_grad():
                for p in (model.wte.weight, model.blocks[0].mlp.c_proj.weight):
                    p.mul_(2.)
            model(x, y).backward()
            assert model.velocity_position_embedding.weight.grad.abs().sum() > 0
            assert model.velocity_embedding.weight.grad.abs().sum() > 0
            opt.step()
    # A never-visited row only saw AdamW decay; projecting the velocity table
    # together with the backbone would incorrectly give it unit length.
    assert model.velocity_position_embedding.weight[-1].float().norm() < .5
