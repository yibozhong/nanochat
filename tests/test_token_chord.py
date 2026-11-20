from dataclasses import replace

import pytest
import torch

from nanochat.chord_triton import fused_gated_write, fused_write
from nanochat.connection_model import ConnectionConfig
from nanochat.lookahead import LookaheadResidual
from nanochat.momentum import normalize_residual, tangent_project
from test_axis_write import test_active_write_cache_and_compiled_training
from test_momentum import gpu, test_exact_optimizer_checkpoint_next_update


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_lookahead', lookahead_mu_init=.3,
                            ngpt_velocity_mode='lookahead_split_learned_beta', token_chord_gates=True,
                            ngpt_force_rms_floor=1e-6)


def test_neutral_gates_preserve_positive_control(setting):
    gated = LookaheadResidual(setting).cuda()
    control = LookaheadResidual(replace(setting, token_chord_gates=False)).cuda()
    h = normalize_residual(torch.randn(32,128,device='cuda'))
    v = tangent_project(h,torch.randn_like(h))
    force = torch.randn_like(h)
    assert torch.equal(gated.gate_slopes,torch.zeros_like(gated.gate_slopes))
    with torch.no_grad():
        control.raw_parallel.add_(.8)
        gated.raw_parallel.copy_(control.raw_parallel)
    actual = gated(h,v,force)
    for x,y in zip(actual,control(h,v,force)):
        torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-5)
    grads, = torch.autograd.grad((actual[0]*torch.randn_like(h)).sum(),gated.gate_slopes)
    assert torch.isfinite(grads).all() and (grads.abs()>0).all()
    # Regression for the original single-step-size Triton entry point.
    with torch.no_grad():
        control.raw_parallel.copy_(control.raw)
    old = fused_write(h,v,force,control.alpha(),.5,1e-6)
    for x,y in zip(old,control(h,v,force)):
        torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-5)


@pytest.mark.parametrize('width,dtype',[(128,torch.float32),(512,torch.bfloat16)])
def test_fused_active_gates_vjp(setting,width,dtype):
    site = LookaheadResidual(setting).cuda()
    with torch.no_grad():
        site.raw_parallel.add_(1.)
        site.raw_beta.add_(.6)
        site.gate_slopes.copy_(torch.tensor([1.8,-2.3],device='cuda'))
    h = normalize_residual(torch.randn(37,width,device='cuda'))
    v = tangent_project(h,torch.randn_like(h))
    f = torch.randn_like(h)
    v[0].zero_()
    f[1].zero_()
    f[2] *= 1e-8
    h[3].fill_(1.)
    f[3].fill_(1.)  # zero chord
    v[3] = tangent_project(h[3],v[3])
    h,v,f = h.requires_grad_(),v.requires_grad_(),f.to(dtype).requires_grad_()
    args = (h,v,f,site.raw,site.raw_parallel,site.raw_beta,site.gate_slopes)
    ref = site.reference_write(h,v,f)
    params = torch.stack((site.alpha(),site.step_size(site.raw_parallel),site.beta(),*site.gate_slopes.unbind()))
    actual = fused_gated_write(h,v,f,params,1e-6)
    for x,y in zip(actual,ref):
        torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-5)
    up = tuple(torch.randn_like(x) for x in ref)
    expected = torch.autograd.grad(ref,args,up)
    observed = torch.autograd.grad(actual,args,up)
    for idx,(x,y) in enumerate(zip(observed,expected)):
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        # Compare degenerate and ordinary rows separately so large force-floor
        # derivatives cannot hide a wrong gate derivative on normal rows.
        groups = ((x[:4],y[:4]),(x[4:],y[4:])) if idx<3 else ((x,y),)
        for a,b in groups:
            err=(a.double()-b.double()).norm()
            assert err <= (.008 if x.dtype==torch.bfloat16 else .0005)*b.double().norm()+1e-5, (idx,err,b.norm())


def test_independent_token_scalings(setting):
    site = LookaheadResidual(setting).cuda().double()
    with torch.no_grad():
        site.gate_slopes.copy_(site.raw.new_tensor([1.3,-2.]))
        site.raw_parallel.add_(.5)
    h = site.raw.new_tensor([[2.,0,0,0]]).expand(2,-1)
    v = site.raw.new_tensor([[0.,1.,0,0]]).expand(2,-1)
    f = site.raw.new_tensor([[0.,1.,1.,0],[0.,-1.,1.,0]])
    delta = normalize_residual(f)-h
    par = (delta*v).sum(-1,True)*v   # v has Euclidean norm one in both rows.
    c = torch.nn.functional.cosine_similarity(delta,v,dim=-1)[:,None]
    a = site.alpha() * (2*torch.sigmoid(site.gate_slopes[0]*c))
    b = site.step_size(site.raw_parallel) * (2*torch.sigmoid(site.gate_slopes[1]*c))
    assert (a>0).all() and (b>0).all() and a[0]>a[1] and b[0]<b[1]
    expected = normalize_residual(h+a*(delta-par)+b*par)
    torch.testing.assert_close(site(h,v,f)[0],expected)
