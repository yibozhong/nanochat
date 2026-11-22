"""Chord can start without a velocity table and acquire velocity after Attention."""
from dataclasses import replace

import pytest
import torch

from nanochat.connection_cost import model_cost
from nanochat.connection_model import ConnectionConfig, build_connection_model
from nanochat.packed_training import seed_all
from test_chord_fused_branch import test_fused_model as check_fused_model
from test_momentum import gpu, inputs, make, test_exact_optimizer_checkpoint_next_update


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256, streams=1,
                            residual_dynamics='ngpt_lookahead', velocity_init='zero',
                            ngpt_velocity_mode='lookahead_split_learned_beta',
                            chord_velocity_geometry='project_only', fuse_chord_read_norm=True,
                            residual_projection_init='normal', ngpt_force_rms_floor=1e-6)


@torch.no_grad()
def test_from_rest_matches_zeroed_table_and_acquires_velocity(setting):
    seed_all(42)
    rest = make(setting).eval()
    seed_all(42)
    control = make(replace(setting, velocity_init='token')).eval()
    assert rest.velocity_embedding is None and rest.velocity_position_embedding is None
    assert set(dict(control.named_parameters())) - set(dict(rest.named_parameters())) == {'velocity_embedding.weight'}
    assert sum(p.numel() for p in control.parameters()) - sum(p.numel() for p in rest.parameters()) == setting.vocab_size*setting.width
    for name, p in rest.named_parameters():
        torch.testing.assert_close(p, control.get_parameter(name), rtol=0, atol=0)
    control.velocity_embedding.weight.zero_()
    ids, _ = inputs()
    h, v = rest.initial_state(rest.wte(ids), ids)
    assert h.dtype == v.dtype == torch.bfloat16 and torch.count_nonzero(v).item() == 0
    first = rest.blocks[0].attn_residual
    assert torch.equal(first.query(h, v), h)
    force = torch.randn_like(h)
    nh, nv = first(h, v, force)
    assert torch.isfinite(nv).all() and torch.count_nonzero(nv).item() > 0
    ref_h, ref_v = first.reference_write(h, v, force)
    torch.testing.assert_close(nh, ref_h, rtol=.01, atol=.008)
    torch.testing.assert_close(nv, ref_v, rtol=.01, atol=.001)
    torch.testing.assert_close(rest(ids), control(ids), rtol=0, atol=0)


def test_fused_training_and_cached_inference(setting):
    check_fused_model(setting, False)


@torch.no_grad()
def test_ffn_capacity_matches_production_table(setting):
    c = replace(setting, depth=8, width=512, heads=4, sequence_length=2048, vocab_size=32768)
    seed_all(42)
    control = build_connection_model(replace(c, velocity_init='token'))
    seed_all(42)
    wide = build_connection_model(replace(c, match_velocity_table_with_ffn=True))
    assert wide.velocity_embedding is None and wide.velocity_position_embedding is None
    assert sum(p.numel() for p in control.parameters()) == sum(p.numel() for p in wide.parameters()) == 75_497_535
    for name, weight in wide.named_parameters():
        original = control.get_parameter(name)
        if name.endswith('mlp.c_fc.weight'):
            assert weight.shape == (4096, 512) and weight[2048:].abs().sum() > 0
            weight = weight[:2048]
        if name.endswith('mlp.c_proj.weight'):
            assert weight.shape == (512, 4096) and weight[:, 2048:].abs().sum() > 0
            weight = weight[:, :2048]
        assert torch.equal(weight, original), name
    new = model_cost(wide.config)
    assert new['velocity_embedding_parameters'] == 0
    assert new['additional_ffn_parameters'] == control.velocity_embedding.weight.numel()
    assert new['estimated_training_flops'] - model_cost(c)['estimated_training_flops'] == (
        6 * c.sequence_length * new['additional_ffn_parameters'])


def test_wide_ffn_training_cache_and_restore(setting, tmp_path):
    # Same 2x FFN expansion as the 10B recipe on a smaller compiled model.
    wide = replace(setting, vocab_size=2048, match_velocity_table_with_ffn=True)
    check_fused_model(wide, False)
    test_exact_optimizer_checkpoint_next_update(wide, tmp_path)
