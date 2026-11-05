"""Compact8 routing formula/VJP, production integration and unchanged defaults."""
from dataclasses import asdict
from pathlib import Path
import json

import pytest
import torch
from torch.nn import functional as F

from nanochat.compact_routing import compact_routing
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for
from nanochat.connection_experiment import configure_attention,match_vanilla_initialization
from nanochat.stream_mlp_connections import StreamMLPHyperConnection
from nanochat.channel_connections import ChannelHyperConnection
from nanochat.packed_training import seed_all


def relative(a,b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-12)).item()


def record(name,value):
    root=Path('results/compact-checks');root.mkdir(parents=True,exist_ok=True)
    (root/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


def reference(x,phi,alpha,beta):
    raw=F.linear(x,phi)/(x.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
    h=(raw*alpha.repeat_interleave(4)+beta).sigmoid()
    return h*torch.tensor([1.]*4+[2.]*4,device=x.device,dtype=x.dtype)


@pytest.mark.parametrize('m,k,amp',[(1,512,1.),(129,2048,1.),(17,512,0.),(17,512,1e-4)])
@pytest.mark.parametrize('tf32',[True,False])
def test_fp64_all_gradients_and_determinism(m,k,amp,tf32):
    seed_all(14);torch.backends.cuda.matmul.allow_tf32=False
    args=[(torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*amp).requires_grad_(),
          (torch.randn(8,k,device='cuda')/k**.5).requires_grad_(),
          torch.tensor([.2,.3],device='cuda',requires_grad=True),
          (torch.randn(8,device='cuda')*.1).requires_grad_()]
    h=compact_routing(*args,use_tf32=tf32);go=torch.randn_like(h)
    grad=torch.autograd.grad(h,args,go)
    oracle=[x.detach().double().requires_grad_() for x in args]
    expected=reference(*oracle);eg=torch.autograd.grad(expected,oracle,go.double())
    errors=dict(output=relative(h,expected),gradients=[relative(a,b) for a,b in zip(grad,eg)])
    record(f'oracle-{m}-{k}-{amp}-{tf32}',errors)
    assert errors['output']<.003 and max(errors['gradients'])<.012,errors
    again=torch.autograd.grad(compact_routing(*args,use_tf32=tf32),args,go)
    assert all(torch.equal(a,b) for a,b in zip(grad,again))


@pytest.mark.parametrize('kind',[StreamMLPHyperConnection,ChannelHyperConnection])
def test_nvidia_read_equivalence_and_all_gradients(kind):
    seed_all(55)
    old=kind(512,4,0).cuda();old.reset_parameters()
    new=kind(512,4,0,compact_routing=True).cuda();new.load_state_dict(old.state_dict())
    x=torch.randn(2,129,512,4,device='cuda',dtype=torch.bfloat16)
    probe=torch.randn(2,129,512,device='cuda',dtype=torch.bfloat16)
    post_probe=torch.randn(2,129,4,device='cuda')
    outputs=[];grads=[]
    for model in (old,new):
        xx=x.detach().requires_grad_();branch,_,post=model.read(xx)
        if isinstance(post,tuple):raise AssertionError('Unexpected mixing path')
        values=[xx,model.phi,model.scales,model.bias]
        grads.append(torch.autograd.grad((branch.float()*probe.float()).sum()+(post*post_probe).sum(),values))
        outputs.append((branch,post))
    errors=dict(output=[relative(a,b) for a,b in zip(outputs[1],outputs[0])],
                gradients=[relative(a,b) for a,b in zip(grads[1],grads[0])])
    record(kind.__name__,errors)
    assert max(errors['output'])<.006 and max(errors['gradients'])<.02,errors


@pytest.mark.parametrize('variant',['mhc_stream_mlp','mhc_channel_no_sinkhorn'])
def test_full_model_initialization_causal_recompute_and_gradients(variant):
    configure_attention('hardware_auto');models=[]
    for compact in (False,True):
        seed_all(42)
        m=ConnectionModel(ConnectionConfig(variant=variant,width=128,depth=2,heads=2,vocab_size=128,compact_routing=compact)).prepare(compile_blocks=True)
        match_vanilla_initialization(m,42)
        with torch.no_grad():
            for block in m.blocks:
                block.attn.c_proj.weight.normal_(std=.03);block.mlp.c_proj.weight.normal_(std=.03)
        models.append(m)
    old,new=models
    assert all(torch.equal(p,new.state_dict()[k]) for k,p in old.state_dict().items())
    ids=torch.randint(0,128,(2,2048),device='cuda');changed=ids.clone();changed[0,1024:]=(changed[0,1024:]+17)%128
    with torch.no_grad():
        a,b=new(ids),new(changed)
        assert torch.equal(a[0,:1024],b[0,:1024]) and torch.equal(a[1],b[1])
    x,y=ids[:,:128].contiguous(),ids[:,1:129].contiguous()
    losses=[];grads=[]
    for m in models:
        loss=m(x,y);loss.backward();losses.append(loss.detach())
        grads.append({k:p.grad.clone() for k,p in m.named_parameters()})
    torch.testing.assert_close(losses[0],losses[1],atol=1e-4,rtol=1e-4)
    errors={k:relative(grads[1][k],grads[0][k]) for k in grads[0]}
    assert all(torch.isfinite(g).all() for g in grads[1].values())
    assert max(errors.values())<.03,errors
    new.zero_grad(set_to_none=True);new.config.activation_checkpoint=True
    loss=new(x,y);loss.backward()
    recompute={k:relative(p.grad,grads[1][k]) for k,p in new.named_parameters()}
    assert max(recompute.values())<.02,recompute
    record('full-'+variant,dict(losses=[v.item() for v in losses],gradients=errors,recompute=recompute))


def test_default_and_checkpoint_schema():
    assert not ConnectionConfig().compact_routing
    for v in ('baseline','mhc','mhc_no_sinkhorn'):
        assert not ConnectionConfig.from_checkpoint(dict(variant=v)).compact_routing
        with pytest.raises(ValueError):ConnectionConfig(variant=v,compact_routing=True)
    c=ConnectionConfig(variant='mhc_stream_mlp',compact_routing=True)
    assert ConnectionConfig.from_checkpoint(asdict(c))==c
