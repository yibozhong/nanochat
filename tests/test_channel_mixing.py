"""Independent FP64 all-path VJP oracle plus actual 2K model invariants."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from nanochat.channel_mixing import channel_mix
from nanochat.channel_connections import ChannelHyperConnection
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for
from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.engine import KVCache
from nanochat.nvidia_ops import nvidia_ops
from nanochat.packed_training import seed_all
from test_connection_training_resume import test_optimizer_rng_checkpoint_and_fixed_gradient_next_update as check_resume

VARIANT='mhc_channel_no_sinkhorn'

def relative(a,b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()

def record(name,value):
    p=Path('results/channel-checks');p.mkdir(parents=True,exist_ok=True)
    (p/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')

def reference(x,y,post,w,b,a):
    v=x/torch.sqrt(x.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps)
    m=b+a*torch.einsum('btcr,cijr->btcij',v,w)
    return torch.einsum('btci,btcij->btcj',x,m)+y.unsqueeze(-1)*post.unsqueeze(-2)

@pytest.mark.parametrize('n,c,amp',[(1,128,1.),(131,128,1.),(129,512,1.),(17,128,0.),(17,128,1e-4)])
def test_fused_all_gradients_against_fp64(n,c,amp):
    torch.manual_seed(209)
    args=[(torch.randn(1,n,c,4,device='cuda',dtype=torch.bfloat16)*amp).requires_grad_(),
          torch.randn(1,n,c,device='cuda',dtype=torch.bfloat16,requires_grad=True),
          torch.randn(1,n,4,device='cuda',requires_grad=True),
          (torch.randn(c,4,4,4,device='cuda')*.5).requires_grad_(),
          (torch.randn(c,4,4,device='cuda')*.2+torch.eye(4,device='cuda')).requires_grad_(),
          torch.tensor([.17],device='cuda',requires_grad=True)]
    out=channel_mix(*args)
    probe=torch.randn_like(out)
    actual=torch.autograd.grad(out,args,probe)
    oracle=[t.detach().double().requires_grad_() for t in args]
    expected=reference(*oracle)
    expected_grad=torch.autograd.grad(expected,oracle,probe.double())
    errors=dict(output=relative(out,expected),gradients=[relative(a,b) for a,b in zip(actual,expected_grad)])
    record(f'oracle-{n}-{c}-{amp}',errors)
    assert errors['output']<.006,errors
    assert max(errors['gradients'])<.012,errors
    assert all(torch.isfinite(g).all() for g in actual)
    again=torch.autograd.grad(channel_mix(*args),args,probe)
    assert all(torch.equal(a,b) for a,b in zip(actual,again))


def test_locality_and_no_saved_coefficient_bank():
    torch.manual_seed(11)
    m=ChannelHyperConnection(128,4,0).cuda();m.reset_parameters()
    x=torch.randn(2,17,128,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.zeros_like(x[...,0]).contiguous();post=torch.zeros(2,17,4,device='cuda')
    saved=[]
    def pack(t):saved.append(tuple(t.shape));return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        out=channel_mix(x,y,post,m.mix_weight,m.mix_bias,m.mix_scale)
    assert len(saved)==6 and (2,17,128,4,4) not in saved,saved
    changed=x.detach().clone();changed[0,3,5]+=1
    after=channel_mix(changed,y,post,m.mix_weight,m.mix_bias,m.mix_scale)
    mask=torch.ones_like(out,dtype=torch.bool);mask[0,3,5]=False
    assert torch.equal(out[mask],after[mask])
    assert not torch.equal(out[0,3,5],after[0,3,5])
    gx,=torch.autograd.grad(out[0,3,5].float().sum(),x)
    assert torch.count_nonzero(gx[mask])==0
    record('locality',dict(saved_shapes=saved,no_cross_token_or_channel_path=True))


def test_pre_post_and_backbone_initialization_unchanged():
    models=[]
    for v in ('mhc_no_sinkhorn',VARIANT):
        seed_all(42)
        model=ConnectionModel(ConnectionConfig(variant=v,depth=2,width=128,heads=2,vocab_size=128)).prepare()
        match_vanilla_initialization(model,42);models.append(model)
    old,new=models
    for name,a in old.named_parameters():
        if '_connection.' not in name:
            assert torch.equal(a,dict(new.named_parameters())[name]),name
    for ob,nb in zip(old.blocks,new.blocks):
        for kind in ('attn_connection','mlp_connection'):
            a,b=getattr(ob,kind),getattr(nb,kind)
            assert torch.equal(a.phi[:8],b.phi)
            assert torch.equal(a.bias[:8],b.bias) and torch.equal(a.scales[:2],b.scales)
            x=torch.randn(1,33,128,4,device='cuda',dtype=torch.bfloat16)
            ar,_,am=a.read(x);br,_,bm=b.read(x)
            assert torch.equal(ar,br) and torch.equal(am[0],bm)
    opt,_=optimizer_for(new)
    routed={id(p) for g in opt.param_groups if g['kind']=='adamw' for p in g['params']}
    assert all(id(p) in routed for n,p in new.named_parameters() if '_connection.' in n)
    assert ConnectionConfig.from_checkpoint(asdict(new.config))==new.config
    assert old.config!=new.config


def test_full_model_causal_cache_cce_recompute_and_no_sinkhorn(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Sinkhorn called in channel variant')
    monkeypatch.setattr(nvidia_ops(),'mhc_fused_sinkhorn',forbidden)
    monkeypatch.setattr(nvidia_ops(),'mhc_generate_mix_and_aggregate',forbidden)
    configure_attention('fa3');seed_all(42)
    cfg=ConnectionConfig(variant=VARIANT,width=128,heads=2,depth=2,vocab_size=128)
    m=ConnectionModel(cfg).prepare(compile_blocks=True)
    with torch.no_grad():
        for b in m.blocks:
            b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
    ids=torch.randint(0,128,(2,2048),device='cuda');changed=ids.clone()
    changed[0,1024:]=(changed[0,1024:]+17)%128
    with torch.no_grad():
        a,b=m(ids),m(changed)
        assert torch.equal(a[0,:1024],b[0,:1024]) and torch.equal(a[1],b[1])
        # Uncompiled branches for variable-length decoding, as in existing tests.
        for block in m.blocks:block.compiled_attention=block.compiled_mlp=None
        cache=KVCache(2,2,32,64,2,'cuda',torch.bfloat16)
        pre=m(ids[:,:7],kv_cache=cache);one=m(ids[:,7:8],kv_cache=cache);full=m(ids[:,:8])
        torch.testing.assert_close(pre,full[:,:7],atol=.004,rtol=.03)
        torch.testing.assert_close(one,full[:,7:],atol=.004,rtol=.03)
    x,y=ids[:,:128].contiguous(),ids[:,1:129].contiguous()
    loss=m(x,y);dense=m(x,y,dense=True)
    torch.testing.assert_close(loss,dense,atol=2e-4,rtol=2e-4)
    loss.backward();grad={k:p.grad.clone() for k,p in m.named_parameters()}
    assert all(torch.isfinite(g).all() for g in grad.values())
    assert all(grad[k].norm()>0 for k in grad if '_connection.' in k)
    m.zero_grad(set_to_none=True);m.config.activation_checkpoint=True
    again=m(x,y);again.backward()
    errors={k:relative(p.grad,grad[k]) for k,p in m.named_parameters()}
    record('model',dict(ce=loss.item(),dense=dense.item(),gradients=errors,sinkhorn_calls=0,causal_tokens=2048))
    torch.testing.assert_close(loss,again,atol=2e-5,rtol=2e-5)
    assert max(errors.values())<.02,errors


def test_optimizer_rng_resume(tmp_path):check_resume(VARIANT,tmp_path)


def test_unsupported_configs_fail_explicitly():
    for opts in (dict(architecture='native'),dict(fused_connection_grad=True)):
        with pytest.raises(ValueError):ConnectionConfig(variant=VARIANT,**opts)
