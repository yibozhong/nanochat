"""Independent all-gradient oracle, identity initialization and 2K integration."""
from dataclasses import asdict
import json
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from nanochat.stream_mlp import stream_mlp
from nanochat.stream_mlp_connections import StreamMLPHyperConnection
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for
from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.engine import KVCache
from nanochat.nvidia_ops import nvidia_ops
from nanochat.packed_training import seed_all
from test_connection_training_resume import test_optimizer_rng_checkpoint_and_fixed_gradient_next_update as check_resume

VARIANT = 'mhc_stream_mlp'


def relative(a,b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()


def record(name,value):
    p=Path('results/stream-mlp-checks');p.mkdir(parents=True,exist_ok=True)
    (p/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


def reference(x,y,post,win,wout):
    return F.linear(F.linear(x,win).relu(),wout)+y.unsqueeze(-1)*post.unsqueeze(-2)


@pytest.mark.parametrize('n,c,amp',[(1,128,1.),(131,128,1.),(129,512,1.),(17,128,0.),(17,128,1e-4)])
def test_all_gradients_against_fp64(n,c,amp):
    seed_all(209)
    args=[(torch.randn(1,n,c,4,device='cuda',dtype=torch.bfloat16)*amp).requires_grad_(),
          torch.randn(1,n,c,device='cuda',dtype=torch.bfloat16,requires_grad=True),
          torch.randn(1,n,4,device='cuda',requires_grad=True),
          (torch.randn(16,4,device='cuda')*.5).requires_grad_(),
          (torch.randn(4,16,device='cuda')*.25).requires_grad_()]
    out=stream_mlp(*args);probe=torch.randn_like(out)
    grads=torch.autograd.grad(out,args,probe)
    oracle=[t.detach().double().requires_grad_() for t in args]
    expected=reference(*oracle)
    expected_grads=torch.autograd.grad(expected,oracle,probe.double())
    errors=dict(output=relative(out,expected),gradients=[relative(a,b) for a,b in zip(grads,expected_grads)])
    record(f'oracle-{n}-{c}-{amp}',errors)
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    assert all(torch.isfinite(g).all() for g in grads)
    again=torch.autograd.grad(stream_mlp(*args),args,probe)
    assert all(torch.equal(a,b) for a,b in zip(grads,again))


def test_identity_initialization_and_no_extra_path():
    seed_all(12);m=StreamMLPHyperConnection(128,4,0).cuda();m.reset_parameters()
    x=torch.randn(2,17,128,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.zeros_like(x[...,0]).contiguous();post=torch.zeros(2,17,4,device='cuda')
    out=m.write(y,x,post)
    assert relative(out,x)<1e-5
    # Both distinct basis pairs participate even on initially identical streams.
    replicated=torch.tensor([[1.,1.,1.,1.],[-1.,-1.,-1.,-1.]],device='cuda')
    hidden=F.linear(replicated,m.mix_in).relu()
    assert (hidden.sum(0)>0).all()
    assert torch.unique(m.mix_in,dim=0).shape[0]==16
    # Check identity derivative only away from ReLU kinks.
    valid=(F.linear(x.float(),m.mix_in).abs()>.01).all(-1)
    gx,=torch.autograd.grad(out.float().sum(-1)[valid].sum(),x)
    expected=valid[...,None].expand_as(x).to(x.dtype)
    assert relative(gx,expected)<1e-5
    with torch.no_grad():
        m.mix_in.normal_(std=.5);m.mix_out.normal_(std=.25)
    torch.testing.assert_close(m.write(y,x*2,post),m.write(y,x,post)*2,atol=0,rtol=0)
    with torch.no_grad():m.mix_out.zero_()
    assert torch.count_nonzero(m.write(y,x,post))==0


def test_locality_sharing_and_no_saved_hidden_bank():
    seed_all(11);m=StreamMLPHyperConnection(128,4,0).cuda();m.reset_parameters()
    with torch.no_grad():m.mix_in.normal_(std=.5);m.mix_out.normal_(std=.25)
    x=torch.randn(2,17,128,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.zeros_like(x[...,0]).contiguous();post=torch.zeros(2,17,4,device='cuda');saved=[]
    def pack(t):saved.append(tuple(t.shape));return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):out=m.write(y,x,post)
    assert len(saved)==5 and (2,17,128,16) not in saved,saved
    changed=x.detach().clone();changed[0,3,5]+=1;after=m.write(y,changed,post)
    mask=torch.ones_like(out,dtype=torch.bool);mask[0,3,5]=False
    assert torch.equal(out[mask],after[mask]) and not torch.equal(out[0,3,5],after[0,3,5])
    gx,=torch.autograd.grad(out[0,3,5].float().sum(),x)
    assert torch.count_nonzero(gx[mask])==0
    copied=x.detach().clone();copied[1,7,111]=copied[0,3,5];mixed=m.write(y,copied,post)
    assert torch.equal(mixed[1,7,111],mixed[0,3,5])
    record('locality',dict(saved_shapes=saved,shared_channel_parameters=True,no_cross_token_or_channel_path=True))


def test_pre_post_backbone_optimizer_and_layer_independence():
    models=[]
    for v in ('mhc_no_sinkhorn',VARIANT):
        seed_all(42);m=ConnectionModel(ConnectionConfig(variant=v,depth=2,width=128,heads=2,vocab_size=128)).prepare()
        match_vanilla_initialization(m,42);models.append(m)
    old,new=models;new_params=dict(new.named_parameters())
    for name,p in old.named_parameters():
        if '_connection.' not in name:assert torch.equal(p,new_params[name]),name
    for ob,nb in zip(old.blocks,new.blocks):
        for kind in ('attn_connection','mlp_connection'):
            a,b=getattr(ob,kind),getattr(nb,kind)
            assert torch.equal(a.phi[:8],b.phi) and torch.equal(a.bias[:8],b.bias) and torch.equal(a.scales[:2],b.scales)
            x=torch.randn(1,33,128,4,device='cuda',dtype=torch.bfloat16)
            ar,_,am=a.read(x);br,_,bm=b.read(x)
            assert torch.equal(ar,br) and torch.equal(am[0],bm)
    def groups(m):
        opt,wd=optimizer_for(m);names={id(p):n for n,p in m.named_parameters()}
        return {names[id(p)]:{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups for p in g['params']},wd
    og,ow=groups(old);ng,nw=groups(new);assert ow==nw
    for name in new_params:
        if '_connection.' in name:assert ng[name]['kind']=='adamw' and ng[name]['weight_decay']==0
        else:assert og[name]==ng[name]
    pointers=[p.data_ptr() for n,p in new.named_parameters() if '_connection.mix_' in n]
    assert len(pointers)==len(set(pointers))==8
    assert ConnectionConfig.from_checkpoint(asdict(new.config))==new.config


def test_full_model_causal_cache_cce_recompute_no_sinkhorn(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('Sinkhorn invoked by stream MLP')
    monkeypatch.setattr(nvidia_ops(),'mhc_fused_sinkhorn',forbidden)
    monkeypatch.setattr(nvidia_ops(),'mhc_generate_mix_and_aggregate',forbidden)
    configure_attention('fa3');seed_all(42)
    m=ConnectionModel(ConnectionConfig(variant=VARIANT,width=128,heads=2,depth=2,vocab_size=128)).prepare(compile_blocks=True)
    with torch.no_grad():
        for b in m.blocks:b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
    ids=torch.randint(0,128,(2,2048),device='cuda');changed=ids.clone();changed[0,1024:]=(changed[0,1024:]+17)%128
    with torch.no_grad():
        a,b=m(ids),m(changed)
        assert torch.equal(a[0,:1024],b[0,:1024]) and torch.equal(a[1],b[1])
        for block in m.blocks:block.compiled_attention=block.compiled_mlp=None
        cache=KVCache(2,2,32,64,2,'cuda',torch.bfloat16)
        pre=m(ids[:,:7],kv_cache=cache);one=m(ids[:,7:8],kv_cache=cache);full=m(ids[:,:8])
        torch.testing.assert_close(pre,full[:,:7],atol=.004,rtol=.03)
        torch.testing.assert_close(one,full[:,7:],atol=.004,rtol=.03)
    x,y=ids[:,:128].contiguous(),ids[:,1:129].contiguous()
    loss=m(x,y);dense=m(x,y,dense=True);torch.testing.assert_close(loss,dense,atol=2e-4,rtol=2e-4)
    loss.backward();grads={k:p.grad.clone() for k,p in m.named_parameters()}
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert all(grads[k].norm()>0 for k in grads if '_connection.' in k)
    m.zero_grad(set_to_none=True);m.config.activation_checkpoint=True
    again=m(x,y);again.backward();errors={k:relative(p.grad,grads[k]) for k,p in m.named_parameters()}
    torch.testing.assert_close(loss,again,atol=2e-5,rtol=2e-5)
    assert max(errors.values())<.02,errors
    record('model',dict(ce=loss.item(),dense=dense.item(),gradients=errors,causal_tokens=2048,sinkhorn_calls=0))


def test_optimizer_rng_resume(tmp_path):check_resume(VARIANT,tmp_path)


def test_config_and_cost():
    from nanochat.connection_cost import model_cost
    for kwargs in (dict(architecture='native'),dict(fused_connection_grad=True)):
        with pytest.raises(ValueError):ConnectionConfig(variant=VARIANT,**kwargs)
    c=ConnectionConfig(variant=VARIANT)
    assert model_cost(c)['components_forward']['stream_mlp']==2*c.depth*c.sequence_length*c.width*256
