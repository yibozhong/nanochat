"""Unit-step spherical HBFormer with unprojected backbone weights."""
from dataclasses import asdict

import pytest
import torch

from nanochat.connection_model import ConnectionConfig, optimizer_for
from nanochat.momentum import MomentumModel
from test_momentum import (
    gpu, inputs, make, test_compiled_checkpoint_forward_backward,
    test_cce_gradient_and_velocity_learning, test_causal_batch_and_cached_chunks,
    test_exact_optimizer_checkpoint_next_update,
)
from test_ngpt_hbformer import test_original_velocity_and_unit_step
from test_nonzero_residual import test_32_layer_production_initialization


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_hbformer', ngpt_weight_norm=False,
                            ngpt_force_rms_floor=1e-6)


def test_backbone_stays_unprojected(setting, monkeypatch):
    stored = asdict(setting)
    assert ConnectionConfig.from_checkpoint(stored).ngpt_weight_norm is False
    del stored['ngpt_weight_norm']
    assert ConnectionConfig.from_checkpoint(stored).ngpt_weight_norm is True

    def unexpected_projection(self):
        pytest.fail('Weight projection was called with ngpt_weight_norm=False')

    monkeypatch.setattr(MomentumModel, 'normalize_backbone_weights', unexpected_projection)
    model = make(setting)
    opt, _ = optimizer_for(model)
    assert .0009 < model.lm_head.weight.std().item() < .0011
    assert .75 < model.wte.weight.float().std().item() < .85
    x, y = inputs()
    model(x, y).backward()
    opt.step()
    assert torch.isfinite(model.lm_head.weight).all()
    assert model.wte.weight.float().norm(dim=1).mean().item() > 2
