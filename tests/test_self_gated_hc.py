"""Scalar-local SiLU, unchanged HC ownership, and real model compatibility."""
from dataclasses import asdict
from pathlib import Path
import json
import pytest
import torch

import nanochat.self_gated_connections as impl
from nanochat.nvidia_connections import NvidiaHyperConnection
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for
from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.packed_training import seed_all
from nanochat.engine import KVCache
from test_connection_training_resume import test_optimizer_rng_checkpoint_and_fixed_gradient_next_update as resume_check

VARIANT='self_gated_hc'


def relative(a,b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()


def record(name,value):
    p=Path('results/self-gate-checks');p.mkdir(parents=True,exist_ok=True)
    (p/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


@pytest.fixture(autouse=True)
def fresh_compile_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize('amplitude',[0.,1e-4,1.,10.])
def test_scalar_vjp_fp64(amplitude):
    seed_all(209)
    x=(torch.randn(2,131,128,4,device='cuda',dtype=torch.bfloat16)*amplitude).requires_grad_()
    probe=torch.randn_like(x);out=impl.self_gate(x);gx,=torch.autograd.grad(out,x,probe)
    xx=x.detach().double().requires_grad_();ref=xx.square()*torch.sigmoid(xx)
    expected,=torch.autograd.grad(ref,xx,probe.double())
    errors=dict(output=relative(out,ref),gradient=relative(gx,expected))
    record('scalar-'+str(amplitude),errors)
    assert errors['output']<.006 and errors['gradient']<.012,errors
    assert torch.isfinite(out).all() and torch.isfinite(gx).all()


def test_scalar_locality_and_no_parameters():
    x=torch.randn(2,17,128,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    before=impl.self_gate(x);changed=x.detach().clone();changed[0,3,7,1]+=2
    after=impl.self_gate(changed);mask=torch.ones_like(x,dtype=torch.bool);mask[0,3,7,1]=False
    assert torch.equal(before[mask],after[mask])
    gx,=torch.autograd.grad(before[0,3,7,1],x)
    assert torch.count_nonzero(gx[mask])==0
    assert before[0,3,7,1]!=after[0,3,7,1]
    record('locality',dict(no_other_token_channel_or_stream_dependency=True,formula='x*silu(x)'))


@pytest.mark.parametrize('n,c',[(1,128),(131,128),(129,512)])
def test_write_all_vjps_fp64(n,c):
    configure_attention('hardware_auto');seed_all(19)
    conn=impl.SelfGatedHyperConnection(c,4,0).cuda();conn.reset_parameters()
    x=torch.randn(1,n,c,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.randn(1,n,c,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    post=torch.randn(1,n,4,device='cuda',requires_grad=True)
    h=(torch.randn(1,n,4,4,device='cuda')*.3+torch.eye(4,device='cuda')).requires_grad_()
    args=[x,y,post,h];probe=torch.randn_like(x)
    out=conn.write(y,x,(post,h,None));actual=torch.autograd.grad(out,args,probe)
    xx,yy,pp,hh=[a.detach().double().requires_grad_() for a in args]
    ref=torch.einsum('btci,btij->btcj',xx.square()*torch.sigmoid(xx),hh)+yy.unsqueeze(-1)*pp.unsqueeze(-2)
    expected=torch.autograd.grad(ref,[xx,yy,pp,hh],probe.double())
    errors=dict(output=relative(out,ref),gradients=[relative(a,b) for a,b in zip(actual,expected)])
    record(f'write-{n}-{c}',errors)
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors


def build(variant=VARIANT):
    configure_attention('hardware_auto');seed_all(42)
    m=ConnectionModel(ConnectionConfig(variant=variant,width=128,heads=2,depth=2,vocab_size=128)).prepare(compile_blocks=True)
    match_vanilla_initialization(m,42)
    return m


def test_unchanged_parameters_routing_rng_optimizer():
    old=build('mhc_no_sinkhorn');old_rng=torch.cuda.get_rng_state()
    new=build();assert torch.equal(old_rng,torch.cuda.get_rng_state())
    assert old.state_dict().keys()==new.state_dict().keys()
    assert all(torch.equal(p,new.state_dict()[name]) for name,p in old.state_dict().items())
    a,b=old.blocks[0].attn_connection,new.blocks[0].attn_connection
    assert isinstance(b,NvidiaHyperConnection) and not b.sinkhorn
    x=torch.randn(2,31,128,4,device='cuda',dtype=torch.bfloat16)
    av,_,am=a.read(x);bv,_,bm=b.read(x)
    assert torch.equal(av,bv) and torch.equal(am[0],bm[0]) and torch.equal(am[1],bm[1])
    opt,_=optimizer_for(new);routed={id(p) for g in opt.param_groups if g['kind']=='adamw' for p in g['params']}
    assert all(id(p) in routed for name,p in new.named_parameters() if '_connection.' in name)
    assert ConnectionConfig.from_checkpoint(asdict(new.config))==new.config
    record('ownership',dict(added_parameters=0,same_rng=True,unchanged_routing=True,identity_initial_gate=False))


def nonzero_model():
    m=build()
    with torch.no_grad():
        for b in m.blocks:b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
    return m


def test_model_causality_cache_cce_recompute():
    m=nonzero_model();ids=torch.randint(0,128,(2,2048),device='cuda');changed=ids.clone()
    changed[0,1024:]=(changed[0,1024:]+17)%128
    with torch.no_grad():
        a,b=m(ids),m(changed)
        assert torch.equal(a[0,:1024],b[0,:1024]) and torch.equal(a[1],b[1])
        compiled=[(b.compiled_attention,b.compiled_mlp) for b in m.blocks]
        for b in m.blocks:b.compiled_attention=b.compiled_mlp=None
        cache=KVCache(2,2,32,64,2,'cuda',torch.bfloat16)
        pre=m(ids[:,:7],kv_cache=cache);one=m(ids[:,7:8],kv_cache=cache);full=m(ids[:,:8])
        torch.testing.assert_close(pre,full[:,:7],atol=.004,rtol=.03)
        torch.testing.assert_close(one,full[:,7:],atol=.004,rtol=.03)
        for b,(attn,mlp) in zip(m.blocks,compiled):b.compiled_attention=attn;b.compiled_mlp=mlp
    x,y=ids[:,:128].contiguous(),ids[:,1:129].contiguous()
    loss=m(x,y);torch.testing.assert_close(loss,m(x,y,dense=True),atol=2e-4,rtol=2e-4)
    loss.backward();grads={n:p.grad.clone() for n,p in m.named_parameters()}
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert all(g.norm()>0 for n,g in grads.items() if '_connection.' in n)
    m.zero_grad(set_to_none=True);m.config.activation_checkpoint=True
    other=m(x,y);other.backward()
    errors={n:relative(p.grad,grads[n]) for n,p in m.named_parameters()}
    assert max(errors.values())<.02,errors
    record('model',dict(loss=loss.item(),recompute_loss=other.item(),gradients=errors))


def test_model_compiled_gate_reference(monkeypatch):
    m=nonzero_model();x=torch.randint(0,128,(2,128),device='cuda');y=(x+1)%128
    loss=m(x,y);loss.backward();grads={n:p.grad.clone() for n,p in m.named_parameters()}
    monkeypatch.setattr(impl,'self_gate',impl.self_gate_reference)
    m.zero_grad(set_to_none=True);other=m(x,y);other.backward()
    errors={n:relative(p.grad,grads[n]) for n,p in m.named_parameters()}
    assert max(errors.values())<.02,errors
    torch.testing.assert_close(loss,other,atol=2e-4,rtol=2e-4)
    record('compiled-reference',dict(gradients=errors,loss=loss.item(),reference=other.item()))


def test_optimizer_rng_resume(tmp_path):
    resume_check(VARIANT,tmp_path,attention_backend='hardware_auto')


@pytest.mark.parametrize('options',[dict(architecture='native'),dict(fused_connection_grad=True),dict(compact_routing=True)])
def test_unsupported_config(options):
    with pytest.raises(ValueError):ConnectionConfig(variant=VARIANT,**options)
