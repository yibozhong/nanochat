"""Vanilla skip scales: formula, all gradients, placement and training lifecycle."""
from dataclasses import asdict
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.skip_scale import scaled_skip
from nanochat.packed_training import seed_all, snapshot, restore_rng
from nanochat.engine import KVCache
from test_connection_training_resume import equal_tree


def relative(a,b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()


def record(name,value):
    root=Path('results/skip-scale-checks');root.mkdir(parents=True,exist_ok=True)
    (root/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


@pytest.fixture(autouse=True)
def fresh_cache():
    torch._dynamo.reset()
    torch.backends.cuda.matmul.allow_tf32=False
    yield
    torch._dynamo.reset()


def build(variant='skip_scale',compiled=False,nonzero=False):
    configure_attention('hardware_auto');seed_all(42)
    m=ConnectionModel(ConnectionConfig(variant=variant,streams=1,depth=2,width=128,
        heads=2,sequence_length=2048,vocab_size=128)).prepare(compile_blocks=compiled)
    match_vanilla_initialization(m,42)
    if nonzero:
        with torch.no_grad():
            for b in m.blocks:
                b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
                if variant=='skip_scale':
                    b.attn_skip_scale.uniform_(.7,1.3);b.mlp_skip_scale.uniform_(.7,1.3)
    return m


@pytest.mark.parametrize('rows,width,mode',[(13,7,'random'),(17,128,'zero'),
    (19,128,'tiny'),(129,512,'signed'),(131,128,'ones'),(32768,512,'random')])
def test_compiled_primitive_all_vjps(rows,width,mode):
    torch.manual_seed(8108)
    x=torch.randn(1,rows,width,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.randn_like(x,requires_grad=True)
    s=torch.randn(width,device='cuda')*.2+1
    if mode=='zero':s.zero_()
    elif mode=='tiny':s.mul_(1e-4)
    elif mode=='signed':s[::2].neg_()
    elif mode=='ones':s.fill_(1)
    s.requires_grad_();args=[x,y,s];go=torch.randn_like(x)
    oracle=[v.detach().double().requires_grad_() for v in args]
    expected=oracle[0]*oracle[2]+oracle[1]
    eg=torch.autograd.grad(expected,oracle,go.double())
    compiled=torch.compile(scaled_skip,fullgraph=True,dynamic=False)
    out=compiled(*args);grads=torch.autograd.grad(out,args,go)
    errors=dict(output=relative(out,expected),gradients=[relative(a,b) for a,b in zip(grads,eg)])
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    assert out.dtype==torch.bfloat16 and grads[2].dtype==torch.float32
    if mode=='ones':assert torch.equal(out,x+y)
    record(f'primitive-{rows}-{width}-{mode}',errors)


def test_identity_parameters_rng_optimizer():
    base=build();rng=torch.get_rng_state()
    vanilla=build('baseline')
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,base.state_dict()[n]) for n,v in vanilla.state_dict().items())
    extra={n:p for n,p in base.named_parameters() if n not in vanilla.state_dict()}
    assert len(extra)==4 and all(p.shape==(128,) and p.dtype==torch.float32 and torch.all(p==1) for p in extra.values())
    assert base.stream_head is None and base.config.streams==1
    assert all(b.attn_connection is None and b.mlp_connection is None for b in base.blocks)
    assert not any('attnres' in n or '_connection.' in n or 'resid_lambdas' in n for n,_ in base.named_parameters())
    # Nonzero branches prevent a vacuous zero-output initialization comparison.
    with torch.no_grad():
        for b in vanilla.blocks:
            b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
        base.load_state_dict(vanilla.state_dict(),strict=False)
        ids=torch.randint(0,128,(2,64),device='cuda')
        assert torch.equal(base(ids),vanilla(ids))
    opt,_=optimizer_for(base)
    scale_groups=[g for g in opt.param_groups if g['kind']=='adamw' and g['weight_decay']==0]
    assert len(scale_groups)==1 and {id(p) for p in scale_groups[0]['params']}=={id(p) for p in extra.values()}
    assert all(p.ndim==2 for g in opt.param_groups if g['kind']=='muon' for p in g['params'])
    assert ConnectionConfig.from_checkpoint(asdict(base.config))==base.config
    with torch.device('meta'):
        full=ConnectionModel(ConnectionConfig(variant='skip_scale',streams=1))
    assert sum(p.numel() for p in full.parameters())==58728448
    record('identity',dict(extra_parameters=8192,total_parameters=58728448,single_stream=True))


def test_only_identity_skip_is_scaled(monkeypatch):
    m=build();b=m.blocks[0];seen=[]
    x=torch.randn(2,11,128,device='cuda',dtype=torch.bfloat16)
    with torch.no_grad():
        b.attn_skip_scale.fill_(2);b.mlp_skip_scale.fill_(3)
    def attention(h,*args):seen.append(h.clone());return h*.25
    def mlp(h):seen.append(h.clone());return h*.5
    monkeypatch.setattr(b,'attention_branch',attention);monkeypatch.setattr(b,'mlp_branch',mlp)
    out=b(x,m.cos[:,:11],m.sin[:,:11])
    intermediate=(x.float()*2+(x*.25).float()).to(x.dtype)
    expected=(intermediate.float()*3+(intermediate*.5).float()).to(x.dtype)
    assert torch.equal(seen[0],x) and torch.equal(seen[1],intermediate) and torch.equal(out,expected)
    a=torch.ones(128,device='cuda');before=scaled_skip(x,torch.zeros_like(x),a)
    a[7]=2;after=scaled_skip(x,torch.zeros_like(x),a)
    mask=torch.ones_like(x,dtype=torch.bool);mask[:,:,7]=False
    assert torch.equal(before[mask],after[mask])


def test_fullmodel_causal_cache_cce_recompute():
    m=build(compiled=True,nonzero=True)
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
    assert all(g.norm()>0 for n,g in grads.items() if n.endswith('_skip_scale'))
    m.zero_grad(set_to_none=True);m.config.activation_checkpoint=True
    again=m(x,y);again.backward()
    errors={n:relative(p.grad,grads[n]) for n,p in m.named_parameters()}
    torch.testing.assert_close(loss,again,atol=2e-5,rtol=2e-5)
    assert max(errors.values())<.02,errors
    record('fullmodel',dict(ce=loss.item(),dense=dense.item(),recompute_gradients=errors))


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


@pytest.mark.parametrize('options',[dict(architecture='native'),dict(fused_connection_grad=True),dict(streams=4)])
def test_rejected_configuration(options):
    opts=dict(variant='skip_scale',streams=1);opts.update(options)
    with pytest.raises(ValueError):ConnectionConfig(**opts)
