"""GPU checks for Chord with an independent velocity token embedding."""
from dataclasses import asdict, replace

import pytest
import torch

from nanochat.connection_model import ConnectionConfig, optimizer_for
from nanochat.momentum import tangent_project
from nanochat.packed_training import seed_all
from test_axis_write import test_active_write_cache_and_compiled_training
from test_momentum import gpu, make, test_exact_optimizer_checkpoint_next_update


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_lookahead', velocity_init='token',
                            residual_projection_init='normal', lookahead_mu_init=.3,
                            ngpt_velocity_mode='lookahead_split_learned_beta', ngpt_force_rms_floor=1e-6,
                            chord_velocity_geometry='project_only')


@torch.no_grad()
def test_only_position_table_removed(setting):
    model = make(setting)
    seed_all(42)
    control = make(replace(setting, velocity_init='token_position'))
    weights, original = dict(model.named_parameters()), dict(control.named_parameters())
    assert set(original) == set(weights) | {'velocity_position_embedding.weight'}
    assert model.velocity_position_embedding is None
    for name, param in weights.items():
        assert torch.equal(param, original[name]), name
    token_weight = model.velocity_embedding.weight
    assert token_weight.data_ptr() != model.wte.weight.data_ptr()
    optimizer, _ = optimizer_for(model)
    groups = [group for group in optimizer.param_groups if any(p is token_weight for p in group['params'])]
    assert len(groups) == 1 and groups[0]['kind'] == 'adamw'

    ids = torch.tensor([[7, 11, 7]], device='cuda')
    h, v = model.initial_state(model.wte(ids), ids)
    torch.testing.assert_close(v, tangent_project(h, model.velocity_embedding(ids).float()), rtol=0, atol=0)
    assert torch.equal(v[:, 0], v[:, 2])
    _, shifted = model.initial_state(model.wte(ids), ids, start=9)
    assert torch.equal(v, shifted)
    assert ConnectionConfig.from_checkpoint(asdict(setting)) == setting
