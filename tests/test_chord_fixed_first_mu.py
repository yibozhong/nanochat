"""Fixed first Attention lookahead in the usual Chord train and inference paths."""
from dataclasses import replace

import pytest
import torch

from nanochat.connection_model import ConnectionConfig, optimizer_for
from test_chord_fused_branch import test_fused_model as check_fused_model
from test_momentum import gpu, inputs, make, test_exact_optimizer_checkpoint_next_update


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256, streams=1,
                            residual_dynamics='ngpt_lookahead', velocity_init='token',
                            ngpt_velocity_mode='lookahead_split_learned_beta',
                            chord_velocity_geometry='project_only', fuse_chord_read_norm=True,
                            ngpt_force_rms_floor=1e-6, fixed_first_attention_mu=True)


def test_fixed_read_and_velocity_embedding_gradient(setting):
    model = make(setting)
    first = model.blocks[0].attn_residual
    sites = [site for block in model.blocks for site in (block.attn_residual, block.mlp_residual)]
    assert first.mu.item() == 1. and not first.mu.requires_grad
    assert all(s.mu.item() == 0. and s.mu.requires_grad for s in sites[1:])
    opt, _ = optimizer_for(model)
    assert all(p is not first.mu for g in opt.param_groups for p in g['params'])

    control = make(replace(setting, fixed_first_attention_mu=False)).blocks[0].attn_residual
    with torch.no_grad():
        control.mu.fill_(1.)
    ids, _ = inputs()
    h, v = model.initial_state(model.wte(ids), ids)
    query = first.branch_input(h, v, h.dtype, fuse_norm=True)
    expected = control.branch_input(h, v, h.dtype, fuse_norm=True)
    torch.testing.assert_close(query, expected, rtol=0, atol=0)
    upstream = torch.randn_like(query)
    grad, = torch.autograd.grad(query, model.velocity_embedding.weight, upstream)
    assert grad[ids.unique()].float().norm() > 0 and torch.isfinite(grad).all()


def test_compiled_training_and_cached_inference(setting):
    check_fused_model(setting, False)
