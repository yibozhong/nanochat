"""Pairwise FFN activation: mathematical oracle, compile, training and cache lifecycle."""
from dataclasses import asdict
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.pairwise_activation import pairwise_activation, pairwise_torch
from nanochat.packed_training import seed_all, snapshot, restore_rng
from nanochat.engine import KVCache
from test_connection_training_resume import equal_tree


def relative(a,b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()


def record(name,value):
    root=Path('results/pairwise-checks');root.mkdir(parents=True,exist_ok=True)
    (root/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


@pytest.fixture(autouse=True)
def fresh_cache():
    torch._dynamo.reset()
    torch.backends.cuda.matmul.allow_tf32=False
    yield
    torch._dynamo.reset()


def build(activation='pairwise',compiled=False,nonzero=False,backend='triton'):
    configure_attention('hardware_auto');seed_all(42)
    m=ConnectionModel(ConnectionConfig(variant='baseline',streams=1,depth=2,width=128,
        heads=2,sequence_length=2048,vocab_size=128,mlp_activation=activation,
        pairwise_backend=backend)).prepare(compile_blocks=compiled)
    match_vanilla_initialization(m,42)
    if nonzero:
        with torch.no_grad():
            for b in m.blocks:
                b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
                if activation=='pairwise':b.mlp.rotation.uniform_(-.3,.3)
    return m


def oracle(x,weight):
    # Independent trig representation of the Cayley transform, evaluated FP64.
    u,v=x.reshape(*x.shape[:-1],-1,2).unbind(-1)
    a=weight/torch.sqrt(1+weight*weight)
    angle=2*torch.atan(a*(u-v)/(1+torch.abs(u)+torch.abs(v)))
    c,s=angle.cos(),angle.sin();p=u.clamp_min(0)**2;q=v.clamp_min(0)**2
    return torch.stack((c*p-s*q,s*p+c*q),-1).flatten(-2)


@pytest.mark.parametrize('rows,width,dtype,mode',[(13,14,'bf16','random'),(17,128,'bf16','zero'),
    (19,130,'bf16','tiny'),(129,512,'bf16','extreme'),(131,128,'bf16','initial'),
    (133,128,'bf16','saturated_weight'),(32768,2048,'bf16','random'),(129,512,'fp32','random')])
def test_primitive_forward_all_vjps(rows,width,dtype,mode):
    torch.manual_seed(8120)
    x=torch.randn(1,rows,width,device='cuda',dtype=torch.bfloat16 if dtype=='bf16' else torch.float32)
    w=torch.randn(width//2,device='cuda')*.25
    if mode=='zero':x.zero_()
    elif mode=='tiny':x.mul_(1e-5)
    elif mode=='extreme':x.mul_(200);w.mul_(4)
    elif mode=='initial':w.zero_()
    elif mode=='saturated_weight':w[::2]=1024;w[1::2]=-1024
    x.requires_grad_();w.requires_grad_();go=torch.randn_like(x)
    reference=[x.detach().double().requires_grad_(),w.detach().double().requires_grad_()]
    expected=oracle(*reference);eg=torch.autograd.grad(expected,reference,go.double())
    before=[a.detach().clone() for a in (x,w)]
    out=torch.compile(pairwise_activation,fullgraph=True,dynamic=False)(x,w)
    grads=torch.autograd.grad(out,(x,w),go)
    error=dict(output=relative(out,expected),gradients=[relative(a,b) for a,b in zip(grads,eg)])
    assert error['output']<(.006 if dtype=='bf16' else 3e-6),error
    assert max(error['gradients'])<(.012 if dtype=='bf16' else 2e-5),error
    assert all(torch.isfinite(a).all() for a in (out,*grads))
    assert torch.equal(x,before[0]) and torch.equal(w,before[1])
    if mode=='initial':assert torch.equal(out,x.relu().square())
    record(f'primitive-{rows}-{width}-{dtype}-{mode}',error)


def test_double_reference_gradcheck():
    torch.manual_seed(51)
    x=torch.randn(2,6,device='cuda',dtype=torch.float64,requires_grad=True)
    w=torch.randn(3,device='cuda',dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(pairwise_torch,(x,w),eps=1e-6,atol=1e-5,rtol=1e-4)
    torch.testing.assert_close(pairwise_torch(x,w),oracle(x,w),atol=1e-12,rtol=1e-12)


def test_pair_norm_content_dependence_and_locality():
    x=torch.randn(2,13,128,device='cuda');w=torch.linspace(-2,2,64,device='cuda')
    out=pairwise_activation(x,w)
    energy=x.relu().square().reshape(2,13,64,2).square().sum(-1)
    actual=out.reshape(2,13,64,2).square().sum(-1)
    torch.testing.assert_close(energy,actual,rtol=2e-6,atol=2e-6)
    changed=x.clone();changed[0,3,2]+=1
    after=pairwise_activation(changed,w);mask=torch.ones_like(x,dtype=torch.bool);mask[0,3,2:4]=False
    assert torch.equal(out[mask],after[mask])
    # Changing an inactive partner changes the active feature's output direction.
    z=torch.tensor([[1.,-1.],[1.,-4.]],device='cuda');weight=torch.ones(1,device='cuda')
    assert not torch.equal(pairwise_activation(z,weight)[0],pairwise_activation(z,weight)[1])


def test_identity_parameters_rng_optimizer():
    new=build();rng=torch.get_rng_state();vanilla=build('relu2')
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,new.state_dict()[n]) for n,v in vanilla.state_dict().items())
    extra={n:p for n,p in new.named_parameters() if n not in vanilla.state_dict()}
    assert len(extra)==2 and all(p.shape==(256,) and p.dtype==torch.float32 and torch.count_nonzero(p)==0 for p in extra.values())
    with torch.no_grad():
        for b in vanilla.blocks:
            b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
        new.load_state_dict(vanilla.state_dict(),strict=False)
        ids=torch.randint(0,128,(2,64),device='cuda')
        assert torch.equal(new(ids),vanilla(ids))
    opt,_=optimizer_for(new);groups=[g for g in opt.param_groups if g['kind']=='adamw' and g['weight_decay']==0]
    assert len(groups)==1 and {id(p) for p in groups[0]['params']}=={id(p) for p in extra.values()}
    assert all(p.ndim==2 for g in opt.param_groups if g['kind']=='muon' for p in g['params'])
    assert ConnectionConfig.from_checkpoint(asdict(new.config))==new.config
    with torch.device('meta'):
        full=ConnectionModel(ConnectionConfig(variant='baseline',streams=1,mlp_activation='pairwise'))
    assert sum(p.numel() for p in full.parameters())==58728448
    record('identity',dict(extra_parameters=8192,total_parameters=58728448))


def test_fullmodel_triton_against_compiled_torch():
    reference=build(compiled=True,nonzero=True,backend='torch');actual=build(compiled=True,nonzero=True)
    equal_tree(reference.state_dict(),actual.state_dict())
    x=torch.randint(0,128,(2,128),device='cuda');y=(x+1)%128
    expected=reference(x,y);expected.backward();observed=actual(x,y);observed.backward()
    errors={n:relative(p.grad,dict(reference.named_parameters())[n].grad) for n,p in actual.named_parameters()}
    torch.testing.assert_close(expected,observed,atol=2e-4,rtol=2e-4)
    assert max(errors.values())<.03,errors
    record('triton-torch',dict(loss_difference=(expected-observed).abs().item(),gradients=errors))


@pytest.mark.parametrize('backend',['triton','torch'])
def test_fullmodel_causal_cache_cce_recompute(backend):
    m=build(compiled=True,nonzero=True,backend=backend)
    ids=torch.randint(0,128,(2,2048),device='cuda');changed=ids.clone()
    changed[0,1024:]=(changed[0,1024:]+17)%128
    with torch.no_grad():
        before,after=m(ids),m(changed)
        assert torch.equal(before[0,:1024],after[0,:1024]) and torch.equal(before[1],after[1])
        compiled=m.compiled_steps;m.compiled_steps=None
        cache=KVCache(2,2,32,64,2,'cuda',torch.bfloat16)
        first=m(ids[:,:7],kv_cache=cache);one=m(ids[:,7:8],kv_cache=cache);full=m(ids[:,:8])
        torch.testing.assert_close(first,full[:,:7],atol=.004,rtol=.03)
        torch.testing.assert_close(one,full[:,7:],atol=.004,rtol=.03)
        m.compiled_steps=compiled
    x,y=ids[:,:128].contiguous(),ids[:,1:129].contiguous()
    loss=m(x,y);dense=m(x,y,dense=True)
    torch.testing.assert_close(loss,dense,atol=2e-4,rtol=2e-4)
    loss.backward();grads={n:p.grad.clone() for n,p in m.named_parameters()}
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert all(g.norm()>0 for n,g in grads.items() if n.endswith('.rotation'))
    m.zero_grad(set_to_none=True);m.config.activation_checkpoint=True
    again=m(x,y);again.backward()
    errors={n:relative(p.grad,grads[n]) for n,p in m.named_parameters()}
    torch.testing.assert_close(loss,again,atol=2e-5,rtol=2e-5)
    assert max(errors.values())<.02,errors
    record('fullmodel-'+backend,dict(ce=loss.item(),dense=dense.item(),recompute_gradients=errors))


def test_fullmodel_compiled_against_eager():
    eager=build(nonzero=True);compiled=build(compiled=True,nonzero=True)
    equal_tree(eager.state_dict(),compiled.state_dict())
    x=torch.randint(0,128,(2,128),device='cuda');y=(x+1)%128
    a=eager(x,y);a.backward();b=compiled(x,y);b.backward()
    errors={n:relative(p.grad,dict(eager.named_parameters())[n].grad) for n,p in compiled.named_parameters()}
    torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
    assert max(errors.values())<.03,errors
    record('eager-compiled',dict(loss_difference=(a-b).abs().item(),gradients=errors))


def test_optimizer_rng_resume(tmp_path):
    m=build(compiled=True);opt,wd=optimizer_for(m)
    x=torch.randint(0,128,(2,128),device='cuda');y=(x+1)%128
    for step in range(4):
        schedule(opt,step,305176,wd);opt.zero_grad(set_to_none=True)
        m(x,y).backward();opt.step()
    gradients={n:p.grad.detach().clone() for n,p in m.named_parameters()}
    path=tmp_path/'checkpoint.pt';torch.save(snapshot(m,opt,4,config=asdict(m.config)),path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    other=build();other_opt,other_wd=optimizer_for(other)
    other.load_state_dict(state['model']);other_opt.load_state_dict(state['optimizer'])
    equal_tree(m.state_dict(),other.state_dict());equal_tree(opt.state_dict(),other_opt.state_dict())
    assert wd==other_wd and state['step']==4
    restore_rng(state['rng']);expected=(torch.rand(8),torch.rand(8,device='cuda'),np.random.rand(8))
    restore_rng(state['rng']);equal_tree(expected,(torch.rand(8),torch.rand(8,device='cuda'),np.random.rand(8)))
    for target,optimizer in ((m,opt),(other,other_opt)):
        for n,p in target.named_parameters():p.grad=gradients[n].clone()
        schedule(optimizer,4,305176,wd);optimizer.step()
    equal_tree(m.state_dict(),other.state_dict());equal_tree(opt.state_dict(),other_opt.state_dict())


def test_actual_attention_forward_backward_backend():
    backend=configure_attention('hardware_auto')
    from nanochat.flash_attention import flash_attn_func
    args=[torch.randn(2,29,2,64,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    go=torch.randn_like(args[0]);torch.autograd.grad(flash_attn_func(*args,causal=True),args,go)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as trace:
        out=flash_attn_func(*args,causal=True);grads=torch.autograd.grad(out,args,go)
    keys=[e.key for e in trace.key_averages()]
    if backend=='fa2':
        assert 'aten::_scaled_dot_product_flash_attention' in keys and 'aten::_scaled_dot_product_flash_attention_backward' in keys
    else:
        assert any('flash' in k.lower() and ('fwd' in k.lower() or 'forward' in k.lower()) for k in keys)
        assert any('flash' in k.lower() and ('bwd' in k.lower() or 'backward' in k.lower()) for k in keys)
    oracle=[a.detach().double().requires_grad_() for a in args];q,k,v=[a.transpose(1,2) for a in oracle]
    mask=torch.ones(29,29,device='cuda',dtype=torch.bool).triu(1)
    expected=((q@k.transpose(-1,-2)/8).masked_fill(mask,float('-inf')).softmax(-1)@v).transpose(1,2)
    eg=torch.autograd.grad(expected,oracle,go.double())
    assert relative(out,expected)<.006 and max(relative(a,b) for a,b in zip(grads,eg))<.012
    record('hardware',dict(backend=backend,operators=keys,gpu=torch.cuda.get_device_name()))


@pytest.mark.parametrize('options',[dict(architecture='native'),dict(fused_connection_grad=True),
    dict(streams=4),dict(attention_kernel='fla_triton'),dict(mlp_activation='bad'),
    dict(pairwise_backend='bad'),dict(variant='mhc')])
def test_rejected_configuration(options):
    opts=dict(variant='baseline',streams=1,mlp_activation='pairwise');opts.update(options)
    with pytest.raises(ValueError):ConnectionConfig(**opts)
