"""Read/write preservation, BF16-boundary FP64 VJPs, and 2K integration."""
from dataclasses import asdict
import json,math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from nanochat.static_connections import StaticHyperConnection
from nanochat.nvidia_connections import NvidiaHyperConnection
from nanochat.nvidia_ops import nvidia_ops
from nanochat.connection_model import ConnectionConfig, ConnectionModel, STATIC_VARIANTS, optimizer_for
from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.engine import KVCache
from nanochat.packed_training import seed_all
from test_connection_training_resume import test_optimizer_rng_checkpoint_and_fixed_gradient_next_update as check_resume


def relative(a,b):return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()
def quant(x):return x+(x.to(torch.bfloat16).double()-x).detach()
def record(name,value):
    p=Path('results/static-checks');p.mkdir(parents=True,exist_ok=True)
    (p/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
def forbid(monkeypatch):
    def failure(*a,**kw):raise AssertionError('Static mixing must not invoke Sinkhorn')
    monkeypatch.setattr(nvidia_ops(),'mhc_fused_sinkhorn',failure)
    monkeypatch.setattr(nvidia_ops(),'mhc_generate_mix_and_aggregate',failure)


@pytest.mark.parametrize('kind',['linear','mlp'])
@pytest.mark.parametrize('width',[128,512])
def test_fp64_all_vjps(kind,width,monkeypatch):
    forbid(monkeypatch);seed_all(42)
    m=StaticHyperConnection(width,4,0,mixing=kind).cuda();m.reset_parameters()
    matrices=[p for n,p in m.named_parameters() if n.startswith('mix_')]
    with torch.no_grad():
        for p in matrices:p.normal_(std=1/math.sqrt(p.shape[1]))
    x=torch.randn(2,17,width,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.randn(2,17,width,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    post=torch.randn(2,17,4,device='cuda',requires_grad=True)
    args=(x,y,post,*matrices)
    m.compiled_mix=torch.compile(m.mix,fullgraph=True,dynamic=False)
    out=m.write(y,x,post);probe=torch.randn_like(out)
    grads=torch.autograd.grad(out,args,probe)
    ref=[p.detach().double().requires_grad_() for p in args];xr,yr,pr,*wr=ref
    flat=xr.flatten(-2)
    if kind=='linear':mixed=quant(F.linear(flat,quant(wr[0])))
    else:mixed=quant(F.linear(quant(F.linear(flat,quant(wr[0]))).relu(),quant(wr[1])))
    expected=quant(mixed.reshape_as(xr)+yr.unsqueeze(-1)*pr.unsqueeze(-2))
    reference=torch.autograd.grad(expected,ref,probe.double())
    errors=dict(output=relative(out,expected),gradients=[relative(a,b) for a,b in zip(grads,reference)],
        oracle='FP64 independent formula, explicit BF16 forward boundaries and straight-through rounding')
    record(f'vjp-{kind}-{width}',errors)
    assert errors['output']<.006 and max(errors['gradients'])<.02,errors
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in grads)


@pytest.mark.parametrize('kind',['linear','mlp'])
def test_no_identity_bypass_no_hidden_norm_and_local_tokens(kind):
    seed_all(12);m=StaticHyperConnection(128,4,0,mixing=kind).cuda();m.reset_parameters()
    x=torch.randn(2,9,128,4,device='cuda',dtype=torch.bfloat16)
    if kind=='linear':assert torch.equal(m.mix(x),x)
    else:
        expected=F.linear(F.linear(x.flatten(-2),m.mix_down.bfloat16()).relu(),m.mix_up.bfloat16()).reshape_as(x)
        assert torch.equal(m.mix(x),expected)
    # Positive homogeneity would fail if an unrequested RMSNorm were added.
    torch.testing.assert_close(m.mix(x*2),m.mix(x)*2,atol=.01,rtol=.01)
    changed=x.clone();changed[0,3]+=2
    a,b=m.mix(x),m.mix(changed)
    mask=torch.ones(2,9,device='cuda',dtype=torch.bool);mask[0,3]=False
    assert torch.equal(a[mask],b[mask])
    assert not torch.equal(a[0,3],b[0,3])
    with torch.no_grad():
        if kind=='linear':m.mix_weight.zero_()
        else:m.mix_up.zero_()
    out=m.write(torch.zeros_like(x[...,0]).contiguous(),x,torch.zeros(2,9,4,device='cuda'))
    assert torch.count_nonzero(out)==0


@pytest.mark.parametrize('variant',STATIC_VARIANTS)
def test_initialization_read_write_and_optimizer(variant):
    models=[]
    for v in ('mhc',variant):
        seed_all(42);m=ConnectionModel(ConnectionConfig(variant=v,width=128,heads=2,depth=2,vocab_size=256)).prepare()
        match_vanilla_initialization(m,42);models.append(m)
    old,new=models;new_params=dict(new.named_parameters())
    for n,p in old.named_parameters():
        if '_connection.' not in n:assert torch.equal(p,new_params[n]),n
    for ob,nb in zip(old.blocks,new.blocks):
        for name in ('attn_connection','mlp_connection'):
            a,b=getattr(ob,name),getattr(nb,name)
            assert torch.equal(a.phi[:8],b.phi) and torch.equal(a.scales[:2],b.scales) and torch.equal(a.bias[:8],b.bias)
            x=torch.randn(1,31,128,4,device='cuda',dtype=torch.bfloat16)
            ar,_,am=a.read(x);br,_,bm=b.read(x)
            assert torch.equal(ar,br) and torch.equal(am[0],bm)
    signatures=[]
    for m in models:
        opt,wd=optimizer_for(m);names={id(p):n for n,p in m.named_parameters()}
        signatures.append(({names[id(p)]:{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups for p in g['params']},wd))
    assert signatures[0][1]==signatures[1][1]
    for n,p in new.named_parameters():
        group=signatures[1][0][n]
        if '_connection.mix_' in n:
            assert p.ndim==2 and group['kind']=='muon' and group['initial_lr']==.005
        elif '_connection.' in n:assert group['kind']=='adamw' and group['weight_decay']==0
        else:assert group==signatures[0][0][n],n
    assert ConnectionConfig.from_checkpoint(asdict(new.config))==new.config
    # Each sublayer owns its own static weights.
    weights=[p.data_ptr() for n,p in new.named_parameters() if '_connection.mix_' in n]
    assert len(weights)==len(set(weights))


@pytest.mark.parametrize('variant',STATIC_VARIANTS)
def test_model_causal_cache_cce_compile_recompute(variant,monkeypatch):
    forbid(monkeypatch);configure_attention('fa3');seed_all(42)
    m=ConnectionModel(ConnectionConfig(variant=variant,depth=2,width=128,heads=2,vocab_size=128)).prepare(compile_blocks=True)
    with torch.no_grad():
        for b in m.blocks:
            b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
    ids=torch.randint(0,128,(2,2048),device='cuda');changed=ids.clone();changed[0,1024:]=(changed[0,1024:]+17)%128
    with torch.no_grad():
        a,b=m(ids),m(changed)
        assert torch.equal(a[0,:1024],b[0,:1024]) and torch.equal(a[1],b[1])
        for b in m.blocks:
            b.compiled_attention=b.compiled_mlp=None
            b.attn_connection.compiled_mix=b.mlp_connection.compiled_mix=None
        cache=KVCache(2,2,32,64,2,'cuda',torch.bfloat16)
        prefix=m(ids[:,:7],kv_cache=cache);one=m(ids[:,7:8],kv_cache=cache);full=m(ids[:,:8])
        torch.testing.assert_close(prefix,full[:,:7],atol=.004,rtol=.03)
        torch.testing.assert_close(one,full[:,7:],atol=.004,rtol=.03)
    x,y=ids[:,:128].contiguous(),ids[:,1:129].contiguous()
    loss=m(x,y);dense=m(x,y,dense=True)
    torch.testing.assert_close(loss,dense,atol=2e-4,rtol=2e-4)
    loss.backward();grads={k:p.grad.clone() for k,p in m.named_parameters()}
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in grads.values())
    m.zero_grad(set_to_none=True);m.config.activation_checkpoint=True
    again=m(x,y);again.backward()
    errors={k:relative(p.grad,grads[k]) for k,p in m.named_parameters()}
    assert max(errors.values())<.02,errors
    record(variant,dict(ce=loss.item(),dense=dense.item(),recompute_gradients=errors,causal_tokens=2048,sinkhorn_calls=0))


@pytest.mark.parametrize('variant',STATIC_VARIANTS)
def test_optimizer_rng_restore(variant,tmp_path):check_resume(variant,tmp_path)


def test_config_and_flops():
    from nanochat.connection_cost import model_cost
    linear=ConnectionConfig(variant='mhc_static_linear');mlp=ConnectionConfig(variant='mhc_static_mlp')
    a,b=model_cost(linear),model_cost(mlp)
    assert a['components_forward']['static_mixing']==2*b['components_forward']['static_mixing']
    for v in STATIC_VARIANTS:
        with pytest.raises(ValueError):ConnectionConfig(variant=v,architecture='native')
        with pytest.raises(ValueError):ConnectionConfig(variant=v,fused_connection_grad=True)
