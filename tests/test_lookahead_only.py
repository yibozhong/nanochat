"""Euclidean lookahead must preserve the vanilla residual at zero gate."""
import pytest
import torch

from nanochat.connection_model import ConnectionConfig
from test_ngpt_lookahead import (
    gpu, inputs, make, test_32_layer_production_initialization,
    test_causal_batch_and_cached_chunks, test_cce_gradient_and_velocity_learning,
    test_compiled_checkpoint_forward_backward, test_exact_optimizer_checkpoint_next_update,
    test_zero_lookahead_matches_baseline_and_learns,
)


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32,
                            vocab_size=256, streams=1, residual_dynamics='lookahead_only',
                            lookahead_mu_init=.3, ngpt_force_rms_floor=1e-6)


def test_affine_read_and_additive_write(setting):
    model = make(setting)
    x, _ = inputs()
    h, v = model.initial_state(model.wte(x), x)
    assert h.dtype == torch.bfloat16
    torch.testing.assert_close(v, model.velocity_embedding(x).float() +
                               model.velocity_position_embedding.weight[:x.shape[1]].float())
    site = model.blocks[0].attn_residual
    assert list(dict(site.named_parameters())) == ['mu']
    # Include a radial component to catch accidental tangent projection.
    v = h.float() * 2
    query = site.query(h, v)
    torch.testing.assert_close(query, h.float() + setting.lookahead_mu_init*v)
    force = torch.randn_like(h)
    new_h, new_v = site(h, v, force)
    assert torch.equal(new_h, h + force) and new_h.dtype == h.dtype
    diff = (h + force).float() - h.float()
    torch.testing.assert_close(new_v, .5*v + .5*diff)
