"""Static input scale: FP64 VJPs, locality, unchanged routing and model invariants."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from nanochat.scaled_mixing import scaled_mix
from nanochat.scaled_connections import ScaledHyperConnection
from nanochat.nvidia_connections import NvidiaHyperConnection
from nanochat.nvidia_ops import nvidia_ops
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for
from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.packed_training import seed_all
from nanochat.engine import KVCache
from test_connection_training_resume import test_optimizer_rng_checkpoint_and_fixed_gradient_next_update as resume_check

VARIANT='scaled_hc'


def reference(x,y,post,h,a):
    return torch.einsum('btci,btij->btcj',x*a,h)+y.unsqueeze(-1)*post.unsqueeze(-2)


def relative(a,b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()


def record(name,value):
    p=Path('results/scaled-hc-checks');p.mkdir(parents=True,exist_ok=True)
    (p/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


@pytest.fixture(autouse=True)
def fresh_compile_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def inputs(n,c,amp=1.,scale_mode='random'):
    a=torch.randn(c,4,device='cuda')*.5+1
    if scale_mode=='zero':a.zero_()
    if scale_mode=='signed':a[:,::2].neg_()
    return [
        (torch.randn(1,n,c,4,device='cuda',dtype=torch.bfloat16)*amp).requires_grad_(),
        torch.randn(1,n,c,device='cuda',dtype=torch.bfloat16,requires_grad=True),
        torch.randn(1,n,4,device='cuda',requires_grad=True),
        (torch.randn(1,n,4,4,device='cuda')*.3+torch.eye(4,device='cuda')).requires_grad_(),
        a.requires_grad_()]


@pytest.mark.parametrize('n,c,amp,mode',[(1,5,1.,'random'),(131,128,1.,'random'),
    (129,512,1.,'random'),(17,128,0.,'random'),(17,128,1e-4,'random'),
    (17,128,1.,'zero'),(17,128,1.,'signed'),(32768,512,1.,'random')])
def test_all_vjps_fp64(n,c,amp,mode):
    torch.manual_seed(209);torch.backends.cuda.matmul.allow_tf32=False
    args=inputs(n,c,amp,mode);go=torch.randn_like(args[0])
    out=scaled_mix(*args);grad=torch.autograd.grad(out,args,go)
    oracle=[a.detach().double().requires_grad_() for a in args]
    expected=reference(*oracle);eg=torch.autograd.grad(expected,oracle,go.double())
    errors=dict(output=relative(out,expected),gradients=[relative(a,b) for a,b in zip(grad,eg)])
    record(f'oracle-{n}-{c}-{amp}-{mode}',errors)
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    assert all(torch.isfinite(g).all() for g in grad)
    again=torch.autograd.grad(scaled_mix(*args),args,go)
    assert all(torch.equal(a,b) for a,b in zip(grad,again))


def test_channel_source_preferences_and_locality():
    args=inputs(3,128)
    x,y,post,h,a=args
    with torch.no_grad():
        x.fill_(1.);y.zero_();a.fill_(1.);h.fill_(1.)
    before=scaled_mix(*args)
    changed=a.detach().clone();changed[7,1]=2.
    after=scaled_mix(x,y,post,h,changed)
    # Scale of source1 affects all four destinations, only channel7.
    assert torch.equal(after[:,:,7]-before[:,:,7],torch.ones_like(after[:,:,7]))
    mask=torch.ones_like(before,dtype=torch.bool);mask[:,:,7]=False
    assert torch.equal(after[mask],before[mask])
    modified=x.detach().clone();modified[0,1,7,1]+=1
    delta=scaled_mix(modified,y,post,h,changed)-after
    mask=torch.ones_like(delta,dtype=torch.bool);mask[0,1,7]=False
    assert torch.count_nonzero(delta[mask])==0
    assert torch.equal(delta[0,1,7],torch.full_like(delta[0,1,7],2.))
    record('channel-source-preferences',dict(static_channel_scale=True,only_own_token=True))


def test_identity_init_routing_rng_optimizer():
    models=[];rng=[]
    for variant in ('mhc_no_sinkhorn',VARIANT):
        seed_all(42)
        m=ConnectionModel(ConnectionConfig(variant=variant,depth=2,width=128,heads=2,vocab_size=128)).prepare()
        match_vanilla_initialization(m,42);models.append(m);rng.append(torch.get_rng_state())
    old,new=models
    assert torch.equal(*rng)
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    oldc,newc=old.blocks[0].attn_connection,new.blocks[0].attn_connection
    assert newc.sinkhorn is False
    assert torch.equal(newc.input_scale,torch.ones_like(newc.input_scale))
    x=torch.randn(2,31,128,4,device='cuda',dtype=torch.bfloat16)
    a,_,ar=oldc.read(x);b,_,br=newc.read(x)
    assert torch.equal(a,b) and torch.equal(ar[0],br[0]) and torch.equal(ar[1],br[1])
    y=torch.randn_like(a)
    torch.testing.assert_close(oldc.write(y,x,ar),newc.write(y,x,br),atol=.03125,rtol=.008)
    opt,_=optimizer_for(new)
    routed={id(p) for g in opt.param_groups if g['kind']=='adamw' for p in g['params']}
    assert all(id(p) in routed for name,p in new.named_parameters() if '_connection.' in name)
    assert ConnectionConfig.from_checkpoint(asdict(new.config))==new.config
    record('identity',dict(initial_scale=1.,same_rng=True))


def test_no_bank_and_compiled_primitive():
    args=inputs(129,128);go=torch.randn_like(args[0]);saved=[]
    def pack(t):saved.append(t.data_ptr());return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        out=scaled_mix(*args)
    assert saved==[a.data_ptr() for a in args],saved
    eg=torch.autograd.grad(out,args,go)
    compiled=torch.compile(scaled_mix,fullgraph=True)(*args)
    actual=torch.autograd.grad(compiled,args,go)
    torch.testing.assert_close(compiled,out,atol=0,rtol=0)
    assert max(relative(a,b) for a,b in zip(actual,eg))<1e-6
    record('no-bank',dict(saved_inputs_only=True,compiled=True))


def model():
    configure_attention('hardware_auto');seed_all(42)
    m=ConnectionModel(ConnectionConfig(variant=VARIANT,width=128,heads=2,depth=2,vocab_size=128)).prepare(compile_blocks=True)
    with torch.no_grad():
        for block in m.blocks:
            block.attn.c_proj.weight.normal_(std=.03);block.mlp.c_proj.weight.normal_(std=.03)
            for conn in (block.attn_connection,block.mlp_connection):conn.input_scale.uniform_(.7,1.3)
    return m


def test_fullmodel_causal_cache_cce_recompute(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Sinkhorn or composite Sinkhorn invoked')
    monkeypatch.setattr(nvidia_ops(),'mhc_fused_sinkhorn',forbidden)
    monkeypatch.setattr(nvidia_ops(),'mhc_generate_mix_and_aggregate',forbidden)
    m=model();ids=torch.randint(0,128,(2,2048),device='cuda');changed=ids.clone()
    changed[0,1024:]=(changed[0,1024:]+17)%128
    with torch.no_grad():
        a,b=m(ids),m(changed)
        assert torch.equal(a[0,:1024],b[0,:1024]) and torch.equal(a[1],b[1])
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
    torch.testing.assert_close(loss,again,atol=2e-5,rtol=2e-5)
    assert max(errors.values())<.02,errors
    record('model',dict(ce=loss.item(),dense=dense.item(),gradients=errors,causal_tokens=2048,sinkhorn_calls=0))


def test_fullmodel_reference_gradients(monkeypatch):
    m=model();x=torch.randint(0,128,(2,128),device='cuda');y=(x+1)%128
    loss=m(x,y);loss.backward();grads={n:p.grad.clone() for n,p in m.named_parameters()}
    def torch_write(self,output,residual,mixes):
        post,h,_=mixes
        return reference(residual.float(),output.float(),post,h,self.input_scale).to(residual.dtype)
    monkeypatch.setattr(ScaledHyperConnection,'write',torch_write)
    m.zero_grad(set_to_none=True);other=m(x,y);other.backward()
    errors={n:relative(grads[n],p.grad) for n,p in m.named_parameters()}
    torch.testing.assert_close(loss,other,atol=2e-4,rtol=2e-4)
    assert max(errors.values())<.02,errors
    record('fullmodel-reference',dict(loss=loss.item(),reference=other.item(),gradients=errors))


def test_optimizer_rng_resume(tmp_path):
    resume_check(VARIANT,tmp_path,attention_backend='hardware_auto')


@pytest.mark.parametrize('options',[dict(architecture='native'),dict(fused_connection_grad=True),dict(compact_routing=True)])
def test_unsupported_config(options):
    with pytest.raises(ValueError):ConnectionConfig(variant=VARIANT,**options)
