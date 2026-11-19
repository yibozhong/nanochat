from dataclasses import replace

import pytest
import torch
from torch.nn.functional import softplus

from nanochat.connection_model import ConnectionConfig, optimizer_for
from nanochat.lookahead import LookaheadResidual
from nanochat.momentum import normalize_residual, tangent_project
from nanochat.packed_training import seed_all
from test_axis_write import test_active_write_cache_and_compiled_training
from test_momentum import gpu, inputs, make, test_exact_optimizer_checkpoint_next_update


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_lookahead', lookahead_mu_init=.3,
                            ngpt_velocity_mode='lookahead_split_write', ngpt_force_rms_floor=1e-6)


def test_equal_steps_and_optimizer_groups(setting):
    baseline = make(replace(setting, ngpt_velocity_mode='lookahead_transport'))
    seed_all(42)
    model = make(setting)
    x,y = inputs()
    assert torch.equal(model(x), baseline(x))
    optimizer,_ = optimizer_for(model)
    model(x,y).backward()
    for b in model.blocks:
        for s in (b.attn_residual,b.mlp_residual):
            assert s.kappa is None and torch.equal(s.raw, s.raw_parallel)
            assert s.alpha().item() == pytest.approx(.05)
            assert s.raw.data_ptr() != s.raw_parallel.data_ptr()
            for p in (s.raw,s.raw_parallel):
                assert torch.isfinite(p.grad).all() and p.grad.abs().item() > 0
                group, = [g for g in optimizer.param_groups if any(q is p for q in g['params'])]
                assert group['kind'] == 'adamw' and group['weight_decay'] == 0


def test_independent_steps_and_projection_gradients(setting):
    s = LookaheadResidual(setting).cuda().double()
    h = normalize_residual(torch.randn(6,19,device='cuda',dtype=torch.float64))
    v = tangent_project(h,torch.randn_like(h)).requires_grad_()
    force = torch.randn_like(h).requires_grad_()
    with torch.no_grad():
        s.raw_parallel.add_(1.)
    a,b = s.alpha(),softplus(s.raw_parallel[0])
    delta = normalize_residual(force)-h
    unit = torch.nn.functional.normalize(v,dim=-1)
    p = (delta*unit).sum(-1,keepdim=True)*unit
    expected = normalize_residual(h + a*(delta-p) + b*p)
    actual,velocity = s(h,v,force)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close((velocity*actual).mean(-1), torch.zeros_like(actual[:,0]))
    probe = torch.randn_like(actual)
    args = (s.raw, s.raw_parallel, v, force)
    observed = torch.autograd.grad((actual*probe).sum(),args)
    wanted = torch.autograd.grad((expected*probe).sum(),args)
    for x,y in zip(observed,wanted):
        torch.testing.assert_close(x,y)
    # At rest the entire write uses the perpendicular step. This also checks
    # for a spurious shared gradient and singular behavior at zero speed.
    rest = torch.zeros_like(v,requires_grad=True)
    actual,vel = s(h,rest,force)
    torch.testing.assert_close(actual,normalize_residual(h+a*delta))
    ga,gb,gv = torch.autograd.grad(((actual+vel)*probe).sum(),(s.raw,s.raw_parallel,rest))
    assert ga.abs().item()>0 and gb.item()==0 and torch.isfinite(gv).all()
