from dataclasses import replace

import pytest
import torch

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
                            ngpt_velocity_mode='lookahead_split_learned_beta', ngpt_force_rms_floor=1e-6)


def test_fixed_beta_initialization_and_optimizer(setting):
    fixed = make(replace(setting, ngpt_velocity_mode='lookahead_split_write'))
    seed_all(42)
    model = make(setting)
    x, y = inputs()
    assert torch.equal(fixed(x), model(x))
    assert model.blocks[-1].mlp_residual.raw_beta is None
    params = [p for n,p in model.named_parameters() if n.endswith('.raw_beta')]
    assert len(params) == 2*setting.depth-1
    assert all(p.item()==0 for p in params)
    opt, _ = optimizer_for(model)
    model(x,y).backward()
    for p in params:
        group, = [g for g in opt.param_groups if any(q is p for q in g['params'])]
        assert group['kind']=='adamw' and group['weight_decay']==0
        assert torch.isfinite(p.grad).all() and p.grad.item()!=0
    opt.step()
    assert all(p.item()!=0 for p in params)


def test_decay_gradient_against_transport(setting):
    site = LookaheadResidual(setting).cuda().double()
    h = normalize_residual(torch.randn(4,16,device='cuda',dtype=torch.double))
    v = tangent_project(h,torch.randn_like(h))
    force, upstream = torch.randn_like(h), torch.randn_like(h)
    with torch.no_grad():
        site.raw_beta.fill_(.8)
        site.raw_parallel.add_(.7)
    beta = site.beta()
    new_h, v_out = site(h,v,force)
    carried = v - ((v*new_h).sum(-1,keepdim=True)/(h.size(-1)+(h*new_h).sum(-1,keepdim=True)))*(h+new_h)
    recent = tangent_project(new_h,new_h-h)
    torch.testing.assert_close(v_out,beta*carried+(1-beta)*recent)
    grad, = torch.autograd.grad((upstream*v_out).sum(),site.raw_beta)
    expected = beta*(1-beta)*((carried-recent)*upstream).sum()
    torch.testing.assert_close(grad[0],expected)
