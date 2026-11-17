"""Nonzero-init, Euclidean Heavy-Ball with only velocity RMS control."""

import pytest
import torch

from nanochat.connection_model import ConnectionConfig
from nanochat.gpt import norm
from test_momentum import (
    gpu, inputs, make, test_backbone_weights_and_optimizer_membership,
    test_compiled_checkpoint_forward_backward, test_cce_gradient_and_velocity_learning,
    test_causal_batch_and_cached_chunks, test_exact_optimizer_checkpoint_next_update,
)
from test_nonzero_residual import test_32_layer_production_initialization


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='heavy_ball_rms', residual_projection_init='normal',
                            ngpt_force_rms_floor=1e-6)


def test_velocity_embedding_and_recurrence(setting):
    model = make(setting, perturb=False)
    x, _ = inputs()
    h, v = model.initial_state(model.wte(x), x)
    embedding = model.velocity_embedding(x).float()
    def reference_rms(t):
        return t / (torch.linalg.vector_norm(t, dim=-1, keepdim=True) / t.shape[-1]**0.5).clamp_min(.1)
    torch.testing.assert_close(v, reference_rms(embedding))
    torch.testing.assert_close(h, norm(model.wte(x)).float())
    update = model.blocks[0].attn_residual
    beta, gamma, eta = update.coefficients()
    assert setting.velocity_init == 'token' and setting.momentum_eta_init == .05
    # A radial force and two magnitudes distinguish raw F from target normalization,
    # tangent projection and hidden retraction. Exercise both sides of the floor.
    for scale in (.001, 2.):
        force = scale*h
        hidden, velocity = update(h, v, force)
        expected = reference_rms(beta*v+gamma*force)
        torch.testing.assert_close(velocity, expected)
        torch.testing.assert_close(hidden, h+eta*expected)
        assert (hidden.square().mean(-1)-1).abs().max() > 1e-4
        v = velocity


def test_velocity_rms_zero_update_gradients(setting):
    update = make(setting).blocks[0].mlp_residual.double()
    h = torch.randn(1,4,device='cuda',dtype=torch.double,requires_grad=True)
    v = torch.zeros_like(h,requires_grad=True)
    f = torch.zeros_like(h,requires_grad=True)
    hidden, velocity = update(h,v,f)
    torch.testing.assert_close(hidden,h)
    assert velocity.count_nonzero()==0
    assert torch.autograd.gradcheck(update, (h,v,f))
