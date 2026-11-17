"""GPU-only geometry, gradient, original-backbone and inference regression gate."""
from dataclasses import asdict, replace

import pytest
import torch

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for
from nanochat.engine import KVCache
from nanochat.packed_training import seed_all, snapshot, restore_rng
from nanochat.riemannian_momentum import momentum_step, parallel_transport, sphere_project, tangent_project
from test_connection_training_resume import equal_tree


@pytest.fixture(autouse=True)
def gpu():
    assert torch.cuda.is_available(), 'Run in the same GPU allocation as the pretraining gate'
    configure_attention('fa2')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    seed_all(42)


def make(compiled=False, checkpoint=False, perturb=True):
    c=ConnectionConfig(depth=2,width=128,heads=1,sequence_length=32,vocab_size=256,
        streams=1,riemannian_momentum=True,activation_checkpoint=checkpoint)
    model=build_connection_model(c).prepare(compile_blocks=compiled)
    if perturb:
        with torch.no_grad():
            for b in model.blocks:
                b.attn.c_proj.weight.normal_(std=.02)
                b.mlp.c_proj.weight.normal_(std=.02)
    return model


def inputs():
    x=torch.randint(0,256,(2,16),device='cuda')
    return x,(x+3)%256


@pytest.mark.parametrize('eta',[0.,.1,1.,100.])
def test_transport_against_independent_rotation_and_inverse(eta):
    h=sphere_project(torch.randn(3,8,device='cuda',dtype=torch.float64))
    u=tangent_project(h,torch.randn_like(h))
    a=tangent_project(h,torch.randn_like(h))
    y=sphere_project(h+eta*u)
    radius=8**.5
    length=u.norm(dim=-1,keepdim=True)
    direction=u/length
    theta=torch.atan(eta*length/radius)
    expected=a+(a*direction).sum(-1,keepdim=True)*((theta.cos()-1)*direction-theta.sin()*h/radius)
    actual=parallel_transport(h,y,a)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close((y*actual).sum(-1),torch.zeros(3,device='cuda',dtype=torch.float64),atol=1e-12,rtol=0)
    torch.testing.assert_close(actual.norm(dim=-1),a.norm(dim=-1),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(parallel_transport(y,h,actual),a,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('zero',[False,True])
def test_complete_geometry_gradcheck_including_zero_velocity(zero):
    h=torch.randn(1,4,device='cuda',dtype=torch.float64,requires_grad=True)
    v=(torch.zeros_like(h) if zero else torch.randn_like(h)).requires_grad_()
    f=(torch.zeros_like(h) if zero else torch.randn_like(h)).requires_grad_()
    coefficients=torch.tensor([.5,.5,1.],device='cuda',dtype=torch.float64,requires_grad=True)
    def fn(x,u,force,c):
        x=sphere_project(x)
        return momentum_step(x,tangent_project(x,u),force,*c.unbind())
    assert torch.autograd.gradcheck(fn,(h,v,f,coefficients),eps=1e-6,atol=2e-6,rtol=1e-4)
    assert torch.autograd.gradgradcheck(fn,(h,v,f,coefficients),eps=1e-6,atol=4e-6,rtol=1e-4)


def test_long_fp32_recurrence_radius_tangency_and_velocity_scale():
    h=sphere_project(torch.randn(4,128,device='cuda'));v=torch.zeros_like(h)
    for i in range(256):
        f=torch.randn_like(h)*(1. if i%2 else .01)
        u=tangent_project(h,.5*v+.5*f)
        y,w=momentum_step(h,v,f,.5,.5,1.)
        torch.testing.assert_close(y.square().mean(-1),torch.ones(4,device='cuda'),atol=5e-7,rtol=0)
        assert ((y*w).sum(-1).abs()/(y.norm(dim=-1)*w.norm(dim=-1)).clamp_min(1e-20)).max()<5e-7
        torch.testing.assert_close(w.norm(dim=-1),u.norm(dim=-1),rtol=5e-7,atol=5e-7)
        h,v=y,w


def test_velocity_is_not_chord_or_eta_scaled_and_zero_force_is_smooth():
    h=sphere_project(torch.randn(2,16,device='cuda',dtype=torch.float64))
    v=tangent_project(h,torch.randn_like(h))
    f=torch.zeros_like(h)
    y,w=momentum_step(h,v,f,.7,.4,2.3)
    torch.testing.assert_close(w.norm(dim=-1),.7*v.norm(dim=-1),rtol=1e-12,atol=1e-12)
    assert not torch.allclose(w,(y-h)/2.3)
    y0,w0=momentum_step(h,torch.zeros_like(h),f,.7,.4,2.3)
    torch.testing.assert_close(y0,h,rtol=1e-12,atol=1e-12)
    assert torch.equal(w0,f)


def test_original_weights_default_and_scalar_optimizer_group():
    m=make(perturb=False)
    seed_all(42)
    base=build_connection_model(replace(m.config,riemannian_momentum=False)).prepare()
    for name,value in base.state_dict().items():
        assert torch.equal(value,m.state_dict()[name]),name
    extra={n:p for n,p in m.named_parameters() if n not in dict(base.named_parameters())}
    assert len(extra)==4 and sum(p.numel() for p in extra.values())==12
    opt,_=optimizer_for(m)
    for g in opt.param_groups:
        if any(id(p) in {id(q) for q in extra.values()} for p in g['params']):
            assert g['kind']=='adamw' and g['weight_decay']==0
    old=asdict(base.config)
    for name in ('riemannian_momentum','momentum_beta_init','momentum_gamma_init','momentum_eta_init'):old.pop(name)
    assert not ConnectionConfig.from_checkpoint(old).riemannian_momentum


@pytest.mark.parametrize('compiled',[False,True])
def test_cce_matches_dense_and_all_parameters_receive_finite_gradient(compiled):
    m=make(compiled);x,y=inputs()
    loss=m(x,y,dense=True);loss.backward()
    expected={n:p.grad.clone() for n,p in m.named_parameters()}
    m.zero_grad(set_to_none=True);other=m(x,y);other.backward()
    torch.testing.assert_close(loss,other,rtol=1e-4,atol=1e-4)
    for n,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),n
        relative=(p.grad.float()-expected[n].float()).norm()/expected[n].float().norm().clamp_min(1e-10)
        assert relative<.035,(n,relative.item())
    # First beta multiplies v0=0, so its zero derivative is intentional.
    assert m.blocks[0].attn_residual.raw.grad[0]==0
    assert m.blocks[-1].mlp_residual.raw.grad.abs().sum()>0


@pytest.mark.parametrize('checkpoint',[False,True])
def test_compiled_and_checkpoint_forward_backward(checkpoint):
    eager=make();seed_all(42);actual=make(compiled=True,checkpoint=checkpoint)
    actual.load_state_dict(eager.state_dict());x,y=inputs()
    a=eager(x,y,dense=True);b=actual(x,y,dense=True)
    a.backward();b.backward()
    torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-4)
    for (n,p),(_,q) in zip(eager.named_parameters(),actual.named_parameters()):
        error=(p.grad.float()-q.grad.float()).norm()/p.grad.float().norm().clamp_min(1e-10)
        assert error<.04,(n,error.item())


def test_causality_and_batch_isolation():
    m=make().eval();x,_=inputs();changed=x.clone();changed[:,7:]=(changed[:,7:]+37)%256
    with torch.no_grad():
        a=m(x);b=m(changed)
        assert torch.equal(a[:,:7],b[:,:7])
        assert not torch.equal(a[:,7:],b[:,7:])
        changed=x.clone();changed[1]=(changed[1]+19)%256
        assert torch.equal(a[0],m(changed)[0])


@torch.no_grad()
def test_prefill_chunks_token_decode_and_cache_reset():
    m=make().eval();x,_=inputs()
    def cache():return KVCache(2,1,32,128,2,'cuda',torch.bfloat16)
    full=m(x);c=cache();prefill=m(x,kv_cache=c)
    c.reset();chunks=torch.cat([m(x[:,:5],kv_cache=c),m(x[:,5:10],kv_cache=c),m(x[:,10:],kv_cache=c)],1)
    c.reset();tokens=torch.cat([m(x[:,i:i+1],kv_cache=c) for i in range(x.shape[1])],1)
    for actual in (prefill,chunks,tokens):
        assert (full-actual).norm()/full.norm()<.025
    assert c.get_pos()==x.shape[1]
    c.reset();torch.testing.assert_close(m(x,kv_cache=c),prefill,rtol=0,atol=0)


def test_checkpoint_optimizer_rng_and_next_update(tmp_path):
    m=make();opt,_=optimizer_for(m);x,y=inputs()
    m(x,y).backward();opt.step();opt.zero_grad(set_to_none=True)
    path=tmp_path/'state.pt';torch.save(snapshot(m,opt,1,config=asdict(m.config),next_data_row=2),path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    restored=build_connection_model(ConnectionConfig.from_checkpoint(state['config'])).prepare()
    other,_=optimizer_for(restored);restored.load_state_dict(state['model']);other.load_state_dict(state['optimizer'])
    equal_tree(m.state_dict(),restored.state_dict());equal_tree(opt.state_dict(),other.state_dict())
    restore_rng(state['rng']);expected=torch.rand(4,device='cuda')
    restore_rng(state['rng']);assert torch.equal(torch.rand(4,device='cuda'),expected)
    m(x,y).backward()
    gradients={n:p.grad.clone() for n,p in m.named_parameters()}
    for model,optimizer in ((m,opt),(restored,other)):
        optimizer.zero_grad(set_to_none=True)
        for n,p in model.named_parameters():p.grad=gradients[n].clone()
        optimizer.step()
    equal_tree(m.state_dict(),restored.state_dict());equal_tree(opt.state_dict(),other.state_dict())


@pytest.mark.parametrize('kwargs',[{'momentum_beta_init':0.},{'momentum_beta_init':1.},
    {'momentum_gamma_init':0.},{'momentum_eta_init':float('nan')},
    {'latent_feedback':True},{'variant':'mhc'},{'architecture':'native'},{'attention_kernel':'fla_triton'}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):ConnectionConfig(riemannian_momentum=True,**kwargs)
