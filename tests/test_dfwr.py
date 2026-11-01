"""Independent inner/outer derivative oracles and causal Transformer integration."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from nanochat.depth_fast_weights import (DepthFastWeights, binding_gradient,
    inner_muon, swiglu, state_memory_estimate)
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for
from nanochat.connection_experiment import configure_attention
from nanochat.packed_training import seed_all
from nanochat.engine import KVCache
import test_connection_training_resume as resume_checks


def record(name,data):
    root=Path('results/dfwr-checks');root.mkdir(exist_ok=True,parents=True)
    (root/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')


def relative(a,b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-15)).item()


def reference_fw(theta,key):
    a,b,c=theta
    gate=torch.einsum('...rd,...d->...r',a,key)
    up=torch.einsum('...rd,...d->...r',b,key)
    return torch.einsum('...dr,...r->...d',c,gate*gate.sigmoid()*up)


def reference_update(theta,momentum,gradients,rates,beta=.95,steps=5):
    weights=[];buffers=[]
    for w,m,g,lr in zip(theta,momentum,gradients,rates):
        m=m*beta+g*(1-beta);z=g*(1-beta)+m*beta
        transposed=z.shape[-2]>z.shape[-1]
        if transposed:z=z.transpose(-2,-1)
        z=z/(torch.linalg.vector_norm(z,dim=(-1,-2),keepdim=True)+1e-7)
        for _ in range(steps):
            a=z@z.transpose(-2,-1)
            z=3.4445*z+(-4.7750*a+2.0315*(a@a))@z
        if transposed:z=z.transpose(-2,-1)
        weights.append(w-lr*max(1,w.shape[-2]/w.shape[-1])**.5*z);buffers.append(m)
    return tuple(weights),tuple(buffers)


def example():
    torch.manual_seed(61)
    theta=tuple((torch.randn(2,3,*s,dtype=torch.float64)*.25).requires_grad_() for s in [(3,6),(3,6),(6,3)])
    momentum=tuple(torch.randn_like(w)*.03 for w in theta)
    key=torch.randn(2,3,6,dtype=torch.float64,requires_grad=True)
    delta=(torch.randn_like(key)*.2).requires_grad_()
    log_rates=torch.full((3,),-4.,dtype=torch.float64,requires_grad=True)
    return theta,momentum,key,delta,log_rates


def test_analytic_inner_and_outer_derivatives():
    theta,momentum,key,delta,logs=example()
    actual,old,loss=binding_gradient(theta,key,delta)
    # Independent theta arguments: key and target are fixed for the INNER
    # derivative, but retain their own paths in the OUTER derivative.
    proxy=tuple(w.clone() for w in theta)
    prediction=reference_fw(proxy,key)
    target=reference_fw(theta,key).detach()+delta
    expected=torch.autograd.grad(.5*(prediction-target).square().sum(),proxy,create_graph=True)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=2e-12,rtol=2e-12)
    wa,ma=inner_muon(theta,momentum,actual,logs.exp())
    wr,mr=reference_update(theta,momentum,expected,logs.exp())
    for a,b in zip((*wa,*ma),(*wr,*mr)):torch.testing.assert_close(a,b,atol=2e-12,rtol=2e-12)
    variables=(*theta,key,delta,logs)
    ga=torch.autograd.grad(swiglu(wa,key)[0].square().sum(),variables,retain_graph=True)
    gr=torch.autograd.grad(reference_fw(wr,key).square().sum(),variables,retain_graph=True)
    for a,b in zip(ga,gr):torch.testing.assert_close(a,b,atol=2e-10,rtol=2e-10)
    assert ga[-2].norm()>0 and ga[-1].norm()>0
    # A tempting -delta simplification has the same VALUE but wrong Hessian.
    bad_proxy=tuple(w.clone() for w in theta)
    bad_output=reference_fw(bad_proxy,key)
    bad=torch.autograd.grad((bad_output*(-delta)).sum(),bad_proxy,create_graph=True)
    bad_weights,_=inner_muon(theta,momentum,bad,logs.exp())
    gb=torch.autograd.grad(swiglu(bad_weights,key)[0].square().sum(),variables)
    mismatch=max(relative(a,b) for a,b in zip(ga,gb))
    assert mismatch>1e-4
    record('derivatives',dict(inner_max_errors=[(a-b).abs().max().item() for a,b in zip(actual,expected)],
        outer_relative_errors=[relative(a,b) for a,b in zip(ga,gr)],wrong_simplification_relative_error=mismatch,
        delta_gradient_norm=ga[-2].norm().item(),log_lr_gradient_norm=ga[-1].norm().item()))


def test_finite_differences_with_frozen_stop_gradient_anchor():
    torch.manual_seed(12)
    theta=tuple((torch.randn(1,*s,dtype=torch.float64)*.3).requires_grad_() for s in [(2,4),(2,4),(4,2)])
    k=torch.randn(1,4,dtype=torch.float64,requires_grad=True)
    d=(torch.randn_like(k)*.2).requires_grad_();rates=torch.full((3,),-.9,dtype=torch.float64,requires_grad=True)
    momentum=tuple(torch.randn_like(w)*.1 for w in theta)
    anchor=reference_fw(theta,k).detach()
    def fn(a,b,c,key,delta,log_lr):
        state=(a,b,c);g,_,_=binding_gradient(state,key,delta,stopped_old=anchor)
        updated,_=inner_muon(state,momentum,g,log_lr.exp())
        return swiglu(updated,key)[0]
    assert torch.autograd.gradcheck(fn,(*theta,k,d,rates),eps=1e-6,atol=2e-5,rtol=2e-4)
    record('finite-differences',dict(passed=True,note='Stop-gradient anchor is frozen under perturbation; recomputing it would test a different derivative.'))


def test_token_state_isolation_momentum_and_functional_updates():
    theta,mom,key,delta,logs=example();original=tuple(w.detach().clone() for w in (*theta,*mom))
    g,_,_=binding_gradient(theta,key,delta)
    updated,velocity=inner_muon(theta,mom,g,logs.exp())
    changed=delta.detach().clone();changed[1,2]+=3
    other,_=inner_muon(theta,mom,binding_gradient(theta,key,changed)[0],logs.exp())
    for a,b in zip(updated,other):
        torch.testing.assert_close(a[0],b[0],atol=0,rtol=0)
        torch.testing.assert_close(a[1,:2],b[1,:2],atol=0,rtol=0)
        assert not torch.equal(a[1,2],b[1,2])
    for a,b in zip((*theta,*mom),original):torch.testing.assert_close(a,b,atol=0,rtol=0)
    next_g=binding_gradient(updated,key,delta)[0]
    carried,_=inner_muon(updated,velocity,next_g,logs.exp())
    reset,_=inner_muon(updated,tuple(torch.zeros_like(w) for w in mom),next_g,logs.exp())
    assert any(not torch.equal(a,b) for a,b in zip(carried,reset))


def test_initial_binding_encodes_embedding():
    torch.manual_seed(9)
    memory=DepthFastWeights(16,8).double()
    h=torch.randn(2,7,16,dtype=torch.float64,requires_grad=True)
    output,theta,momentum=memory.initialize(h)
    error=relative(output,h)
    assert error<1e-4,error
    torch.testing.assert_close(swiglu(theta,memory.key(h))[0],output,atol=0,rtol=0)
    assert all(torch.count_nonzero(m)==0 for m in momentum)
    assert all(w.shape[:2]==(2,7) for w in theta)
    assert torch.autograd.grad(output.square().sum(),h)[0].norm()>0
    record('initial-binding',dict(relative_reconstruction_error=error,no_hidden_bypass=True))


def test_memory_estimate_and_config_validation():
    full=state_memory_estimate(1,2048,512,512,8)
    small=state_memory_estimate(1,2048,512,8,8)
    assert full['live_state_bytes']==12*1024**3
    assert small['live_state_bytes']==192*1024**2
    with pytest.raises(ValueError):ConnectionConfig(variant='dfwr',architecture='native')
    with pytest.raises(ValueError):ConnectionConfig(variant='dfwr',fw_bottleneck=0)
    record('memory-estimate',dict(full=full,bottleneck8=small))


def build(**kwargs):
    configure_attention('sdpa_cudnn');torch.backends.cuda.matmul.allow_tf32=False
    seed_all(42)
    options=dict(depth=2,width=128,heads=2,vocab_size=256,sequence_length=2048,fw_bottleneck=4)
    options.update(kwargs)
    return build_connection_model(ConnectionConfig(variant='dfwr',**options)).prepare()


def test_no_residual_and_exactly_two_updates_per_block(monkeypatch):
    m=build(sequence_length=32);calls=[]
    def forbidden(*args,**kwargs):raise AssertionError('Conventional residual block was called')
    for b in m.blocks:monkeypatch.setattr(b,'forward',forbidden)
    def read_zero(h,delta,theta,momentum):
        calls.append(tuple(h.shape));return torch.zeros_like(h),theta,momentum
    monkeypatch.setattr(m.depth_memory,'update_read',read_zero)
    idx=torch.randint(0,256,(1,16),device='cuda')
    assert torch.count_nonzero(m(idx))==0
    assert len(calls)==2*m.config.depth


def test_real_2k_causality_batch_isolation_cache_and_state_reset():
    configure_attention('sdpa_cudnn');m=build()
    idx=torch.randint(0,256,(2,2048),device='cuda');changed=idx.clone()
    changed[0,1024:]=(changed[0,1024:]+7)%256;changed[1]=(changed[1]+13)%256
    before={k:v.clone() for k,v in m.state_dict().items()}
    with torch.no_grad():
        a,b=m(idx),m(changed)
        torch.testing.assert_close(a[0,:1024],b[0,:1024],atol=0,rtol=0)
        torch.testing.assert_close(a,m(idx),atol=0,rtol=0)
        cache=KVCache(1,2,32,64,2,'cuda',torch.bfloat16)
        pre=m(idx[:1,:7],kv_cache=cache);one=m(idx[:1,7:8],kv_cache=cache);full=m(idx[:1,:8])
        torch.testing.assert_close(pre,full[:,:7],atol=.005,rtol=.04)
        torch.testing.assert_close(one,full[:,7:],atol=.005,rtol=.04)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert not any('runtime' in k or 'momentum' in k for k in m.state_dict())
    record('causality',dict(context=2048,future_and_other_sequence_isolated=True,reset=True,cache=True))


def test_outer_ntp_reaches_all_slow_parameters_and_cce_matches_dense():
    configure_attention('sdpa_cudnn');m=build(sequence_length=128)
    x=torch.randint(0,256,(1,128),device='cuda');y=(x+1)%256
    actual=m(x,y);actual.backward()
    gradients={n:p.grad.clone() for n,p in m.named_parameters()}
    m.zero_grad(set_to_none=True);expected=m(x,y,dense=True);expected.backward()
    torch.testing.assert_close(actual,expected,atol=2e-4,rtol=2e-4)
    errors={n:relative(g,m.get_parameter(n).grad) for n,g in gradients.items()}
    norms={n:g.float().norm().item() for n,g in gradients.items()}
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in gradients.values()),norms
    # Compare the global gradient, avoiding ill-conditioned tiny coordinates.
    error=relative(torch.cat([g.flatten().float() for g in gradients.values()]),
        torch.cat([p.grad.flatten().float() for p in m.parameters()]))
    assert error<.03,error
    record('outer-ntp',dict(cce=actual.item(),dense=expected.item(),global_gradient_relative_error=error,
        parameter_gradient_norms=norms,parameter_gradient_relative_errors=errors))


def test_compiled_checkpointed_gradients():
    configure_attention('sdpa_cudnn');m=build(sequence_length=128)
    x=torch.randint(0,256,(1,128),device='cuda');y=(x+1)%256
    eager=m(x,y);eager.backward();eg=torch.cat([p.grad.flatten().float() for p in m.parameters()])
    m.zero_grad(set_to_none=True);m.prepare(compile_blocks=True)
    compiled=m(x,y);compiled.backward();cg=torch.cat([p.grad.flatten().float() for p in m.parameters()])
    m.zero_grad(set_to_none=True);m.config.activation_checkpoint=True
    recomputed=m(x,y);recomputed.backward();rg=torch.cat([p.grad.flatten().float() for p in m.parameters()])
    torch.testing.assert_close(eager,compiled,atol=3e-4,rtol=3e-4)
    torch.testing.assert_close(compiled,recomputed,atol=3e-4,rtol=3e-4)
    assert relative(cg,eg)<.03
    assert relative(rg,cg)<.02
    record('compile-checkpoint',dict(compiled_relative_error=relative(cg,eg),recomputed_relative_error=relative(rg,cg)))


def test_optimizer_ownership_and_config_roundtrip():
    m=build();opt,_=optimizer_for(m)
    owner={id(p):g for g in opt.param_groups for p in g['params']}
    for n,p in m.depth_memory.named_parameters():
        assert owner[id(p)]['kind']==('adamw' if n=='log_learning_rates' else 'muon')
    assert owner[id(m.depth_memory.log_learning_rates)]['weight_decay']==0
    assert ConnectionConfig.from_checkpoint(asdict(m.config))==m.config


def test_exact_optimizer_rng_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(resume_checks,'build',lambda ignored:build(sequence_length=128,heads=1))
    resume_checks.test_optimizer_rng_checkpoint_and_fixed_gradient_next_update('baseline',tmp_path)
