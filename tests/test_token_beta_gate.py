from dataclasses import replace

import pytest
import torch

from nanochat.connection_model import ConnectionConfig, optimizer_for
from nanochat.lookahead import LookaheadResidual
from nanochat.momentum import normalize_residual, tangent_project
from test_axis_write import test_active_write_cache_and_compiled_training
from test_momentum import gpu, inputs, make, test_exact_optimizer_checkpoint_next_update


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_lookahead', lookahead_mu_init=.3,
                            velocity_init='token_position', ngpt_velocity_mode='lookahead_split_learned_beta',
                            token_beta_gate=True, ngpt_force_rms_floor=1e-6)


def test_neutral_gate_and_optimizer(setting):
    model = make(setting)
    control = make(replace(setting, token_beta_gate=False))
    shared = {k:v for k,v in model.state_dict().items() if not k.endswith('.beta_gate_slope')}
    control.load_state_dict(shared)
    x, y = inputs()
    loss = model(x,y)
    torch.testing.assert_close(loss, control(x,y), rtol=2e-4, atol=2e-4)
    loss.backward()
    opt, _ = optimizer_for(model)
    slopes = [p for n,p in model.named_parameters() if n.endswith('.beta_gate_slope')]
    assert len(slopes) == 2*setting.depth-1 and model.blocks[-1].mlp_residual.beta_gate_slope is None
    for p in slopes:
        group, = [g for g in opt.param_groups if any(q is p for q in g['params'])]
        assert group['kind'] == 'adamw' and group['weight_decay'] == 0
        assert p.item() == 0 and torch.isfinite(p.grad).all() and p.grad.item() != 0
    opt.step()
    assert all(p.item() != 0 for p in slopes)


def test_token_alignment_changes_only_velocity(setting):
    site = LookaheadResidual(setting).cuda().double()
    h = site.raw.new_tensor([[2.,0.,0.,0.]]).expand(2,-1)
    v = site.raw.new_tensor([[0.,1.,0.,0.],[0.,-1.,0.,0.]])
    force = site.raw.new_tensor([[0.,2.,0.,0.]]).expand_as(h)
    original, _ = site(h,v,force)
    with torch.no_grad():
        site.beta_gate_slope.fill_(1.7)
    new_h, velocity = site(h,v,force)
    torch.testing.assert_close(new_h, original)
    # These two tokens travel the same arc, with opposite incoming velocities.
    # The transported direction is the oriented 90-degree rotation of the endpoint.
    direction = torch.stack((-new_h[:,1], new_h[:,0], new_h[:,2], new_h[:,3]), dim=-1)/2
    sign = v[:,1:2]
    recent = tangent_project(new_h, new_h-h)
    beta = (site.raw_beta + sign*site.beta_gate_slope).sigmoid()
    torch.testing.assert_close(velocity, beta * sign * direction + (1-beta)*recent)
    torch.testing.assert_close((new_h*velocity).sum(-1), torch.zeros_like(beta[:,0]), atol=1e-12, rtol=0)


@pytest.mark.parametrize('width,dtype', [(128,torch.float32), (512,torch.bfloat16)])
def test_active_fused_vjp(setting, width, dtype):
    site = LookaheadResidual(setting).cuda()
    with torch.no_grad():
        site.raw_parallel.add_(.8)
        site.raw_beta.add_(.45)
        site.beta_gate_slope.fill_(-1.3)
    h = normalize_residual(torch.randn(37,width,device='cuda'))
    v = tangent_project(h,torch.randn_like(h))
    f = torch.randn_like(h)
    v[0].zero_()  # zero history
    h[1:3].fill_(1.)
    v[1:3,::2], v[1:3,1::2] = 1., -1.
    f[1].fill_(1.)  # exact zero chord
    f[2].zero_()     # force normalization floor
    f[3].mul_(1e-8)
    h, v, f = h.requires_grad_(), v.requires_grad_(), f.to(dtype).requires_grad_()
    args = (h, v, f, site.raw, site.raw_parallel, site.raw_beta, site.beta_gate_slope)
    reference = site.reference_write(h,v,f)
    fused = site(h,v,f)
    for a,b in zip(fused,reference):
        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-5)
    upstream = tuple(torch.randn_like(t) for t in reference)
    expected = torch.autograd.grad(reference,args,upstream)
    observed = torch.autograd.grad(fused,args,upstream)
    for i,(a,b) in enumerate(zip(observed,expected)):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        groups = [(a[j],b[j]) for j in range(4)] + [(a[4:],b[4:])] if i < 3 else [(a,b)]
        for actual,wanted in groups:
            err = (actual.double()-wanted.double()).norm()
            assert err <= (.008 if a.dtype==torch.bfloat16 else .0005)*wanted.double().norm()+1e-5, (i,err,wanted.norm())
