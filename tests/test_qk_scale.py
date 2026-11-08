"""Normalized QK channel weights: pair geometry, gradients, cache and recovery."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.qk_scale import scale_query, QKScaledAttention
from nanochat.gpt import norm, apply_rotary_emb
from nanochat.packed_training import seed_all, snapshot, restore_rng
from nanochat.engine import KVCache
from test_connection_training_resume import equal_tree


def relative(a,b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()


def record(name,value):
    root=Path('results/qk-scale-checks');root.mkdir(parents=True,exist_ok=True)
    (root/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


@pytest.fixture(autouse=True)
def fresh_cache():
    torch._dynamo.reset();torch.backends.cuda.matmul.allow_tf32=False
    yield
    torch._dynamo.reset()


def build(variant='qk_scale',compiled=False,nonzero=False):
    configure_attention('hardware_auto');seed_all(42)
    m=ConnectionModel(ConnectionConfig(variant=variant,streams=1,depth=2,width=128,
        heads=2,sequence_length=2048,vocab_size=128)).prepare(compile_blocks=compiled)
    match_vanilla_initialization(m,42)
    if nonzero:
        with torch.no_grad():
            for b in m.blocks:
                b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
                if variant=='qk_scale':b.attn.qk_scale.uniform_(.7,1.3)
    return m


@pytest.mark.parametrize('rows,heads,half,mode',[(13,2,7,'random'),(17,2,32,'zero'),
    (19,2,32,'tiny'),(129,4,64,'signed'),(131,2,32,'ones'),(32768,4,64,'random')])
def test_compiled_primitive_all_vjps(rows,heads,half,mode):
    torch.manual_seed(8128)
    q=torch.randn(1,rows,heads,2*half,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    s=torch.randn(heads,half,device='cuda')*.2+1
    if mode=='zero':s.zero_()
    elif mode=='tiny':s.mul_(1e-4)
    elif mode=='signed':s[:,::2].neg_()
    elif mode=='ones':s.fill_(1)
    s.requires_grad_();go=torch.randn_like(q)
    oq=q.detach().double().requires_grad_();os=s.detach().double().requires_grad_()
    expected=oq*torch.cat((os,os),dim=-1)[None,None]
    eg=torch.autograd.grad(expected,[oq,os],go.double())
    compiled=torch.compile(scale_query,fullgraph=True,dynamic=False)
    out=compiled(q,s);grads=torch.autograd.grad(out,[q,s],go)
    errors=dict(output=relative(out,expected),gradients=[relative(a,b) for a,b in zip(grads,eg)])
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    assert out.dtype==torch.bfloat16 and grads[1].dtype==torch.float32
    if mode=='ones':assert torch.equal(out,q)
    record(f'primitive-{rows}-{heads}-{half}-{mode}',errors)


def test_identity_parameters_rng_optimizer():
    m=build();rng=torch.get_rng_state();vanilla=build('baseline')
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,m.state_dict()[n]) for n,v in vanilla.state_dict().items())
    extra={n:p for n,p in m.named_parameters() if n not in vanilla.state_dict()}
    assert len(extra)==2 and all(n.endswith('.attn.qk_scale') and p.shape==(2,32) and p.dtype==torch.float32 and torch.all(p==1) for n,p in extra.items())
    assert m.stream_head is None and m.config.streams==1
    assert all(b.attn_connection is None and b.mlp_connection is None for b in m.blocks)
    assert not any('attnres' in n or '_connection.' in n or 'skip_scale' in n or 'resid_lambdas' in n for n,_ in m.named_parameters())
    with torch.no_grad():
        for b in vanilla.blocks:
            b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
        m.load_state_dict(vanilla.state_dict(),strict=False)
        ids=torch.randint(0,128,(2,64),device='cuda')
        assert torch.equal(m(ids),vanilla(ids))
    opt,_=optimizer_for(m)
    groups=[g for g in opt.param_groups if g['kind']=='adamw' and g['weight_decay']==0]
    assert len(groups)==1 and {id(p) for p in groups[0]['params']}=={id(p) for p in extra.values()}
    assert ConnectionConfig.from_checkpoint(asdict(m.config))==m.config
    with torch.device('meta'):full=ConnectionModel(ConnectionConfig(variant='qk_scale',streams=1))
    assert sum(p.numel() for p in full.parameters())==58722304
    record('identity',dict(extra_parameters=2048,total_parameters=58722304,single_stream=True))


def test_postnorm_placement_and_untouched_kv(monkeypatch):
    m=build();attn=m.blocks[0].attn
    with torch.no_grad():attn.qk_scale.copy_(torch.linspace(.5,1.5,64,device='cuda').reshape(2,32))
    x=torch.randn(2,17,128,device='cuda',dtype=torch.bfloat16);cos,sin=m.cos[:,:17],m.sin[:,:17]
    q=attn.c_q(x).view(2,17,2,64);k=attn.c_k(x).view(2,17,2,64);v=attn.c_v(x).view(2,17,2,64)
    q=norm(apply_rotary_emb(q,cos,sin))*1.2;k=norm(apply_rotary_emb(k,cos,sin))*1.2
    expected=(q.float()*torch.cat((attn.qk_scale,attn.qk_scale),dim=-1)).to(q.dtype)
    seen={}
    def capture(a,b,c,**kwargs):
        seen.update(q=a,k=b,v=c,kwargs=kwargs);return c
    import nanochat.qk_scale as module
    monkeypatch.setattr(module.flash_attn,'flash_attn_func',capture)
    out=attn(x,None,(cos,sin),(-1,0),None)
    assert torch.equal(seen['q'],expected) and torch.equal(seen['k'],k) and torch.equal(seen['v'],v)
    assert seen['kwargs']['causal'] and torch.equal(out,attn.c_proj(v.reshape(2,17,128)))
    base=scale_query(q,torch.ones_like(attn.qk_scale));changed=torch.ones_like(attn.qk_scale);changed[1,7]=2
    delta=scale_query(q,changed)-base
    mask=torch.ones_like(delta,dtype=torch.bool);mask[:,:,1,7]=mask[:,:,1,39]=False
    assert torch.count_nonzero(delta[mask])==0


def test_rope_pair_common_shift_invariance():
    torch.manual_seed(51)
    q=torch.randn(2,13,2,64,device='cuda',dtype=torch.float64)
    k=torch.randn(2,17,2,64,device='cuda',dtype=torch.float64)
    s=torch.randn(2,32,device='cuda',dtype=torch.float64)
    frequency=1/(100000**(torch.arange(0,64,2,device='cuda',dtype=torch.float64)/64))
    def rotate(x,offset):
        freq=(torch.arange(x.shape[1],device='cuda',dtype=torch.float64)+offset)[:,None]*frequency
        return apply_rotary_emb(x,freq.cos()[None,:,None,:],freq.sin()[None,:,None,:])
    def scores(offset):
        return scale_query(norm(rotate(q,offset)),s).transpose(1,2)@norm(rotate(k,offset)).transpose(1,2).transpose(-1,-2)
    torch.testing.assert_close(scores(0),scores(37),rtol=1e-11,atol=1e-11)
    torch.testing.assert_close(scale_query(rotate(q,7),s),rotate(scale_query(q,s),7),rtol=1e-12,atol=1e-12)


def test_weighted_attention_and_all_vjps_fp64():
    configure_attention('hardware_auto');torch.manual_seed(82)
    q,k,v=[torch.randn(2,17,2,64,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    s=(torch.randn(2,32,device='cuda')*.2+1).requires_grad_();go=torch.randn_like(q)
    from nanochat.flash_attention import flash_attn_func
    out=flash_attn_func(scale_query(q,s),k,v,causal=True)
    gradients=torch.autograd.grad(out,[q,k,v,s],go)
    oq,ok,ov,os=[x.detach().double().requires_grad_() for x in (q,k,v,s)]
    qs=oq*torch.cat((os,os),-1)[None,None]
    score=qs.transpose(1,2)@ok.transpose(1,2).transpose(-1,-2)/8
    mask=torch.ones(17,17,device='cuda',dtype=torch.bool).triu(1)
    expected=(score.masked_fill(mask,float('-inf')).softmax(-1)@ov.transpose(1,2)).transpose(1,2)
    oracle=torch.autograd.grad(expected,[oq,ok,ov,os],go.double())
    errors=dict(output=relative(out,expected),gradients=[relative(a,b) for a,b in zip(gradients,oracle)])
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    record('weighted-attention',errors)


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
    assert all(g.norm()>0 for n,g in grads.items() if n.endswith('.qk_scale'))
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
    opts=dict(variant='qk_scale',streams=1);opts.update(options)
    with pytest.raises(ValueError):ConnectionConfig(**opts)


def test_retained_skip_scale_identity_optimizer():
    import test_skip_scale as retained
    retained.test_identity_parameters_rng_optimizer()
