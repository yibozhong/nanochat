"""Tangent/scale/transport invariants plus production model regressions, GPU only."""
from dataclasses import asdict, replace

import pytest
import torch

from nanochat.connection_model import ConnectionConfig
from nanochat.momentum import (normalize_residual, tangent_project, sphere_transport,
                              transported_momentum_step)
from scripts.connection_momentum_debug import residual_audit
from test_momentum import (gpu, inputs, make,
    test_backbone_weights_and_optimizer_membership,
    test_compiled_checkpoint_forward_backward,
    test_cce_gradient_and_velocity_learning,
    test_causal_batch_and_cached_chunks,
    test_exact_optimizer_checkpoint_next_update)


@pytest.fixture(params=['token', 'zero'])
def setting(request):
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32,
        vocab_size=256, streams=1, residual_dynamics='ngpt_heavy_ball',
        velocity_init=request.param, residual_projection_init='normal', ngpt_force_rms_floor=1e-6)


@pytest.mark.parametrize('theta', [0., .01, .7, 1.5])
def test_parallel_transport_matches_independent_rotation(theta):
    d = 16
    h = normalize_residual(torch.randn(d, device='cuda', dtype=torch.float64))
    a = h/d**.5
    e = torch.randn_like(a); e = e-(e@a)*a; e = e/e.norm()
    angle = h.new_tensor(theta); c, s = angle.cos(), angle.sin()
    new_h = d**.5*(c*a+s*e)
    # Dense reference for tests only: rotate the great-circle plane, leaving
    # its orthogonal complement unchanged. Production never creates d x d.
    rot = torch.eye(d, device='cuda', dtype=h.dtype) + (c-1)*(torch.outer(a,a)+torch.outer(e,e)) + s*(torch.outer(e,a)-torch.outer(a,e))
    vectors = tangent_project(h, torch.randn(3,d,device='cuda',dtype=h.dtype))
    moved = sphere_transport(h, new_h, vectors)
    torch.testing.assert_close(moved, vectors@rot.T, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(moved@moved.T, vectors@vectors.T, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close((moved*new_h).sum(-1), torch.zeros(3,device='cuda',dtype=h.dtype), atol=2e-12,rtol=0)
    torch.testing.assert_close(sphere_transport(new_h,h,moved), vectors, atol=2e-12,rtol=2e-12)


def test_step_matches_independent_dense_projection_and_transport():
    d=16
    h=normalize_residual(torch.randn(d,device='cuda',dtype=torch.float64))
    v, f = [torch.randn_like(h) for _ in range(2)]
    beta,gamma,eta=h.new_tensor([.5,.4,.07]).unbind()
    actual, carried=transported_momentum_step(h,v,f,beta,gamma,eta,1e-6,.1)
    projection=torch.eye(d,device=h.device,dtype=h.dtype)-torch.outer(h,h)/d
    z=f/f.norm()*d**.5
    u=projection@(beta*v+gamma*z)
    speed=u/(u.norm()/d**.5).clamp_min(.1)
    expected=h+eta*speed; expected=expected/expected.norm()*d**.5
    transport=torch.eye(d,device=h.device,dtype=h.dtype)-torch.outer(h+expected,expected)/(d+h@expected)
    torch.testing.assert_close(actual,expected,atol=2e-12,rtol=2e-12)
    torch.testing.assert_close(carried,transport@speed,atol=2e-12,rtol=2e-12)
    assert not torch.allclose(carried, (actual-h)/eta)


@pytest.mark.parametrize('scale', [0., .001, 1.])
def test_first_second_derivatives_through_projection_norm_and_transport(scale):
    h=torch.randn(1,4,device='cuda',dtype=torch.float64,requires_grad=True)
    v=(scale*torch.randn_like(h)).requires_grad_()
    f=(scale*torch.randn_like(h)).requires_grad_()
    coeff=h.new_tensor([.6,.4,.05],requires_grad=True)
    def fn(h,v,f,coeff):
        return transported_momentum_step(normalize_residual(h),v,f,*coeff.unbind(),.1,.1)
    assert torch.autograd.gradcheck(fn,(h,v,f,coeff),eps=1e-6,atol=2e-6,rtol=1e-4)
    assert torch.autograd.gradgradcheck(fn,(h,v,f,coeff),eps=1e-6,atol=5e-6,rtol=2e-4)


def test_zero_identity_radial_rejection_and_scale_control():
    h=normalize_residual(torch.randn(3,128,device='cuda',dtype=torch.float64))
    b,g,e=h.new_tensor([.5,.5,.05]).unbind()
    for f,v in [(torch.zeros_like(h),torch.zeros_like(h)), (h,3*h)]:
        y,w=transported_momentum_step(h,v,f,b,g,e,1e-6,.1)
        torch.testing.assert_close(y,h,atol=2e-12,rtol=2e-12)
        torch.testing.assert_close(w,torch.zeros_like(w),atol=2e-12,rtol=0)
    v,f=[torch.randn_like(h) for _ in range(2)]
    y,w=transported_momentum_step(h,v,f,b,g,e,1e-6,.1)
    # A common positive rescale of the mixed velocity has no effect above floor.
    yy,ww=transported_momentum_step(h,v,f,2*b,2*g,e,1e-6,.1)
    torch.testing.assert_close(y,yy,atol=2e-12,rtol=2e-12)
    torch.testing.assert_close(w,ww,atol=2e-12,rtol=2e-12)
    torch.testing.assert_close(w.square().mean(-1),torch.ones(3,device=h.device,dtype=h.dtype),atol=2e-12,rtol=0)


def test_initial_embedding_projection_and_scale(setting):
    model=make(setting,perturb=False);x,_=inputs()
    for scale in [1.,1e5]:
        if model.velocity_embedding is not None:
            with torch.no_grad():model.velocity_embedding.weight.mul_(scale)
        h,v=model.initial_state(model.wte(x),x)
        assert v.square().mean(-1).max() < 1+2e-6
        assert (h*v).mean(-1).abs().max() < 2e-6
        if setting.velocity_init=='zero':assert torch.count_nonzero(v)==0
        else:assert v.abs().sum()>0


def test_256_updates_bounded_tangent_speed_with_large_steps():
    h=normalize_residual(torch.randn(4,128,device='cuda'))
    v=normalize_residual(tangent_project(h,torch.randn_like(h)),.1)
    for i in range(256):
        scale=[0.,1e-8,1.,1e4][i%4]
        f=scale*torch.randn_like(h)
        h,v=transported_momentum_step(h,v,f,h.new_tensor(.999),h.new_tensor(10.),
                                     h.new_tensor(10. if i%3==0 else .05),1e-6,.1)
        assert torch.isfinite(h).all() and torch.isfinite(v).all()
        assert (h.square().mean(-1)-1).abs().max()<1e-6
        assert v.square().mean(-1).max()<1+2e-6
        assert (h*v).mean(-1).abs().max()<2e-6


def test_32_layer_gradients_and_geometry(setting):
    model=make(replace(setting,depth=32),perturb=False); x,y=inputs()
    loss=model(x,y,dense=True);loss.backward()
    assert torch.isfinite(loss)
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert len(residual_audit(model,x,require_unit_target=True))==64


def test_new_defaults_and_explicit_legacy_checkpoint():
    current=ConnectionConfig(residual_dynamics='ngpt_heavy_ball')
    assert current.velocity_init=='token' and current.ngpt_velocity_mode=='transport_rms'
    assert current.momentum_eta_init==.05 and current.momentum_velocity_rms_floor==.1
    assert ConnectionConfig.from_checkpoint(asdict(current))==current
    old=asdict(current);old.pop('ngpt_velocity_mode');old.pop('momentum_velocity_rms_floor')
    old['momentum_eta_init']=.1
    restored=ConnectionConfig.from_checkpoint(old)
    assert restored.ngpt_velocity_mode=='ambient' and restored.momentum_eta_init==.1
    for fields in ({'ngpt_velocity_mode':'bad'}, {'momentum_velocity_rms_floor':0},
                   {'momentum_velocity_rms_floor':float('nan')}):
        with pytest.raises(ValueError):ConnectionConfig(**fields)


@torch.no_grad()
def test_pre_revision_checkpoint_keeps_ambient_outputs():
    c=ConnectionConfig(depth=2,width=128,heads=1,sequence_length=32,vocab_size=256,
                       streams=1,residual_dynamics='ngpt_heavy_ball',ngpt_velocity_mode='ambient',
                       residual_projection_init='normal',ngpt_force_rms_floor=1e-6)
    old=asdict(c);old.pop('ngpt_velocity_mode');old.pop('momentum_velocity_rms_floor')
    original=make(c,perturb=False).eval()
    restored=make(ConnectionConfig.from_checkpoint(old),perturb=False).eval()
    restored.load_state_dict(original.state_dict())
    x,_=inputs()
    torch.testing.assert_close(original(x),restored(x),rtol=0,atol=0)

