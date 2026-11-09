"""SWA-only local recurrence, independent full layers, and single MLP per layer."""
from dataclasses import asdict, replace
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.connection_experiment import configure_attention
from nanochat.connection_cost import model_cost
from nanochat.gpt import norm
from nanochat.packed_training import seed_all, snapshot, restore_rng
from nanochat.engine import KVCache
from test_connection_training_resume import equal_tree


def relative(a, b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()


def record(name, value):
    root=Path('results/swa-looping-checks');root.mkdir(parents=True,exist_ok=True)
    (root/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


@pytest.fixture(autouse=True)
def fresh_cache():
    torch._dynamo.reset()
    torch.backends.cuda.matmul.allow_tf32=False
    yield
    torch._dynamo.reset()


def config(**options):
    values=dict(variant='baseline',streams=1,depth=4,width=128,heads=2,
                sequence_length=2048,vocab_size=128,swa_looping=True,swa_window=512,swa_loops=3)
    values.update(options)
    return ConnectionConfig(**values)


def build(compiled=False,nonzero=False,**options):
    configure_attention('fa3');seed_all(42)
    m=ConnectionModel(config(**options)).prepare(compile_blocks=compiled)
    if nonzero:
        with torch.no_grad():
            for b in m.blocks:
                b.attn.c_proj.weight.normal_(std=.03)
                b.mlp.c_proj.weight.normal_(std=.03)
    return m


def test_schedule_parameters_rng_and_optimizer():
    m=build();rng=torch.get_rng_state()
    base=build(swa_looping=False)
    assert torch.equal(rng,torch.get_rng_state())
    equal_tree(m.state_dict(),base.state_dict())
    assert m.config.attention_schedule==((512,3),(512,3),(512,3),(-1,1))
    assert m.stream_head is None
    assert all(b.attn.ve_gate is None and b.attn_connection is None and b.mlp_connection is None for b in m.blocks)
    ids=[id(p) for b in m.blocks for p in b.parameters()]
    assert len(ids)==len(set(ids)), 'No sharing between distinct layer positions'
    opt,wd=optimizer_for(m);base_opt,base_wd=optimizer_for(base)
    equal_tree(opt.state_dict(),base_opt.state_dict());assert wd==base_wd
    assert ConnectionConfig.from_checkpoint(asdict(m.config))==m.config
    with pytest.raises(ValueError,match='retired attention looping'):
        ConnectionConfig.from_checkpoint(dict(asdict(m.config),attention_looping=True))
    legacy=asdict(base.config)
    for key in ('swa_looping','swa_loops','swa_window'):legacy.pop(key)
    assert not ConnectionConfig.from_checkpoint(legacy).swa_looping
    with torch.device('meta'):
        full=ConnectionModel(config(depth=8,width=512,heads=4,vocab_size=32768))
    assert sum(p.numel() for p in full.parameters())==58720256
    assert sum(n for w,n in full.config.attention_schedule if w>=0)==18
    assert sum(n for w,n in full.config.attention_schedule if w<0)==2
    record('ownership',dict(parameters=58720256,extra_parameters=0,swa_evaluations=18,
                           full_evaluations=2,mlp_evaluations=8))


@pytest.mark.parametrize('passes',[1,2,3])
def test_call_order_refined_hidden_and_single_mlp(passes):
    m=build(nonzero=True,swa_loops=passes,swa_window=7)
    events=[];handles=[]
    for i,b in enumerate(m.blocks):
        def attention_hook(module,args,i=i):
            events.append((i,'attn',args[3],args[0].detach().clone()))
        def mlp_hook(module,args,i=i):
            events.append((i,'mlp',None,args[0].detach().clone()))
        handles.extend([b.attn.register_forward_pre_hook(attention_hook),b.mlp.register_forward_pre_hook(mlp_hook)])
    ids=torch.randint(0,128,(1,23),device='cuda')
    out=m.hidden(ids)
    for h in handles:h.remove()
    x=norm(m.wte(ids));cos,sin=m.cos[:,:23],m.sin[:,:23]
    j=0
    for i,b in enumerate(m.blocks):
        left,count=(7,passes) if i<3 else (-1,1)
        for _ in range(count):
            event=events[j];j+=1
            assert event[:3]==(i,'attn',(left,0))
            assert torch.equal(event[3],norm(x))
            x=x+b.attn(norm(x),None,(cos,sin),(left,0),None)
        assert events[j][:3]==(i,'mlp',None) and torch.equal(events[j][3],norm(x));j+=1
        x=x+b.mlp(norm(x))
    assert j==len(events) and torch.equal(out,norm(x))
    if passes>1:assert not torch.equal(events[0][3],events[1][3])


def test_shared_gradients_equal_sum_of_untied_passes():
    m=build(nonzero=True,swa_window=7);b=m.blocks[0]
    copies=[copy.deepcopy(b.attn) for _ in range(3)];mlp=copy.deepcopy(b.mlp)
    x=torch.randn(2,31,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    other=x.detach().clone().requires_grad_();go=torch.randn_like(x)
    cos,sin=m.cos[:,:31],m.sin[:,:31]
    actual=b(x,cos,sin);actual.backward(go)
    expected=other
    for a in copies:expected=expected+a(norm(expected),None,(cos,sin),(7,0),None)
    expected=expected+mlp(norm(expected));expected.backward(go)
    assert torch.equal(actual,expected) and torch.equal(x.grad,other.grad)
    errors={}
    for name,p in b.attn.named_parameters():
        total=sum(dict(a.named_parameters())[name].grad for a in copies)
        errors[name]=relative(p.grad,total)
    assert max(errors.values())<1e-5,errors
    for (name,p),(name2,q) in zip(b.mlp.named_parameters(),mlp.named_parameters()):
        assert name==name2 and torch.equal(p.grad,q.grad)
    record('shared-gradients',errors)


@pytest.mark.parametrize('length,left',[(31,7),(641,512),(2048,512)])
def test_real_windowed_fa3_all_vjps(length,left):
    configure_attention('fa3');seed_all(1042)
    from nanochat.flash_attention import flash_attn_func
    args=[torch.randn(1,length,1,64,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    go=torch.randn_like(args[0])
    out=flash_attn_func(*args,causal=True,window_size=(left,0))
    grads=torch.autograd.grad(out,args,go)
    oracle=[a.detach().double().requires_grad_() for a in args]
    q,k,v=[a.transpose(1,2) for a in oracle]
    pos=torch.arange(length,device='cuda');distance=pos[:,None]-pos[None,:]
    mask=(distance>=0)&(distance<=left)
    expected=((q@k.transpose(-1,-2)/8).masked_fill(~mask,float('-inf')).softmax(-1)@v).transpose(1,2)
    eg=torch.autograd.grad(expected,oracle,go.double())
    errors=dict(output=relative(out,expected),gradients=[relative(a,b) for a,b in zip(grads,eg)])
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    record(f'fa3-{length}-{left}',errors)


def test_window_support_and_loop_reach():
    configure_attention('fa3')
    from nanochat.flash_attention import flash_attn_func
    # Constant q/k and a single source impulse make the mask's exact support visible.
    q=torch.zeros(1,31,1,64,device='cuda',dtype=torch.bfloat16)
    v=torch.zeros_like(q);v[:,2]=1
    for step in range(1,4):
        v=flash_attn_func(q,q,v,causal=True,window_size=(3,0))
        support=(v[0,:,0,0]!=0).nonzero().flatten().tolist()
        assert support==list(range(2,2+3*step+1)),support


def test_all_full_schedule_matches_unchanged_baseline():
    m=build(nonzero=True,window_pattern='L');base=build(nonzero=True,swa_looping=False)
    equal_tree(m.state_dict(),base.state_dict())
    x=torch.randint(0,128,(2,65),device='cuda');y=(x+1)%128
    a=m(x,y);a.backward();b=base(x,y);b.backward()
    assert torch.equal(a,b)
    for (_,p),(_,q) in zip(m.named_parameters(),base.named_parameters()):assert torch.equal(p.grad,q.grad)


def test_fullmodel_2k_causal_cce_recompute_and_cache_rejection():
    m=build(compiled=True,nonzero=True)
    ids=torch.randint(0,128,(2,2048),device='cuda');changed=ids.clone()
    changed[0,1024:]=(changed[0,1024:]+17)%128
    with torch.no_grad():
        before,after=m(ids),m(changed)
        assert torch.equal(before[0,:1024],after[0,:1024]) and torch.equal(before[1],after[1])
        cache=KVCache(2,2,2048,64,4,'cuda',torch.bfloat16)
        pos=cache.get_pos()
        with pytest.raises(ValueError,match='separate per-pass KV caches'):m(ids[:,:7],kv_cache=cache)
        assert cache.get_pos()==pos
    x,y=ids[:,:129].contiguous(),ids[:,1:130].contiguous()
    loss=m(x,y);dense=m(x,y,dense=True)
    torch.testing.assert_close(loss,dense,atol=2e-4,rtol=2e-4)
    loss.backward();grads={n:p.grad.clone() for n,p in m.named_parameters()}
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in grads.values())
    m.zero_grad(set_to_none=True);m.config.activation_checkpoint=True
    again=m(x,y);again.backward()
    errors={n:relative(p.grad,grads[n]) for n,p in m.named_parameters()}
    torch.testing.assert_close(loss,again,atol=2e-5,rtol=2e-5)
    assert max(errors.values())<.02,errors
    record('fullmodel',dict(ce=loss.item(),dense=dense.item(),recompute_gradients=errors))


def test_compiled_against_eager():
    eager=build(nonzero=True);compiled=build(compiled=True,nonzero=True)
    equal_tree(eager.state_dict(),compiled.state_dict())
    x=torch.randint(0,128,(2,129),device='cuda');y=(x+1)%128
    a=eager(x,y);a.backward();b=compiled(x,y);b.backward()
    errors={n:relative(p.grad,dict(eager.named_parameters())[n].grad) for n,p in compiled.named_parameters()}
    torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
    assert max(errors.values())<.03,errors
    record('eager-compiled',dict(loss_difference=(a-b).abs().item(),gradients=errors))


def test_optimizer_rng_resume(tmp_path):
    m=build(compiled=True);opt,wd=optimizer_for(m)
    x=torch.randint(0,128,(2,129),device='cuda');y=(x+1)%128
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


def test_matrix_cost_independent_enumeration():
    c=config(depth=8,width=512,heads=4,vocab_size=32768)
    for passes in (1,3):
        test=replace(c,swa_loops=passes);t,d=test.sequence_length,test.width
        pairs=sum(min(i+1,513) for i in range(t))
        attn_pairs=6*passes*pairs+2*sum(range(1,t+1))
        projection=2*t*d*d*(4*(6*passes+2)+8*8)
        expected=projection+4*d*attn_pairs+2*t*d*test.vocab_size
        assert model_cost(test)['estimated_forward_flops']==expected
    assert model_cost(replace(c,window_pattern='L'))['estimated_forward_flops']==model_cost(replace(c,swa_looping=False))['estimated_forward_flops']


@pytest.mark.parametrize('options',[dict(architecture='native'),dict(variant='mhc_no_sinkhorn'),
    dict(streams=4),dict(attention_kernel='fla_triton'),dict(swa_loops=0),dict(swa_loops=True),dict(swa_window=-1)])
def test_rejected_configuration(options):
    with pytest.raises(ValueError):config(**options)
