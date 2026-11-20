from dataclasses import replace

import pytest
import torch

from nanochat.connection_model import ConnectionConfig
from nanochat.lookahead import LookaheadResidual
from nanochat.momentum import normalize_residual, tangent_project
from nanochat.packed_training import seed_all
from test_axis_write import test_active_write_cache_and_compiled_training
from test_momentum import gpu, inputs, make, test_exact_optimizer_checkpoint_next_update


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_lookahead', lookahead_mu_init=.3,
                            ngpt_velocity_mode='lookahead_split_learned_beta', signed_chord_steps=True,
                            ngpt_force_rms_floor=1e-6)


def test_matched_initial_steps_and_derivatives(setting):
    positive = make(replace(setting, signed_chord_steps=False))
    seed_all(42)
    signed = make(setting)
    x, _ = inputs()
    assert torch.equal(positive(x), signed(x))
    for model in (positive, signed):
        site = model.blocks[0].attn_residual
        for raw in (site.raw, site.raw_parallel):
            value = site.step_size(raw)
            slope, = torch.autograd.grad(value, raw)
            assert value.item() == pytest.approx(.05)
            assert slope.item() == pytest.approx(.0487705755)
    assert signed.residual_metrics() == positive.residual_metrics()


def test_both_signed_components_and_gradients(setting):
    site = LookaheadResidual(setting).cuda().double()
    h = normalize_residual(torch.randn(12,16,device='cuda',dtype=torch.double))
    v = tangent_project(h,torch.randn_like(h))
    force, probe = torch.randn_like(h), torch.randn_like(h)
    delta = normalize_residual(force)-h
    axis = torch.nn.functional.normalize(v,dim=-1)
    parallel = (delta*axis).sum(-1,keepdim=True)*axis
    perp = delta-parallel
    for a,b in ((-.1,.2),(.2,-.1),(-.1,-.2),(0.,0.)):
        for raw, value in ((site.raw,a),(site.raw_parallel,b)):
            if value <= site.initial_alpha:
                raw_value = site.initial_raw + (value-site.initial_alpha)/site.initial_slope
            else:
                raw_value = torch.log(torch.expm1(h.new_tensor(value))).item()
            with torch.no_grad():
                raw.fill_(raw_value)
        new_h,velocity = site(h,v,force)
        torch.testing.assert_close(new_h,normalize_residual(h+a*perp+b*parallel))
        torch.testing.assert_close((new_h*velocity).mean(-1),torch.zeros_like(h[:,0]))
        gradients = torch.autograd.grad((new_h*probe).sum(),(site.raw,site.raw_parallel))
        assert all(torch.isfinite(g).all() and g.abs().item()>0 for g in gradients)
