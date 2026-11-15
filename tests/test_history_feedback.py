"""GPU checks for strict-past sequence feedback and fixed-pass KV caching."""
import copy
from dataclasses import asdict,replace
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from nanochat.connection_model import ConnectionConfig,build_connection_model,optimizer_for
from nanochat.connection_cost import model_cost
from nanochat.gpt import apply_rotary_emb,norm
from nanochat.history_feedback import history_read
from nanochat.packed_training import seed_all
from test_latent_feedback import gpu,inputs
from test_projected_feedback import relative
from test_connection_training_resume import equal_tree


def make(backend='flash',compiled=False,detach=True,checkpoint=False,perturb=True,query='embedding'):
    seed_all(42)
    c=ConnectionConfig(depth=2,width=128,heads=2,sequence_length=32,vocab_size=256,
        streams=1,latent_feedback=True,feedback_source='history_qkv',feedback_attention_backend=backend,
        feedback_detach=detach,activation_checkpoint=checkpoint,feedback_query=query)
    m=build_connection_model(c).prepare(compile_blocks=compiled)
    if perturb:
        with torch.no_grad():
            for b in m.blocks:
                b.attn.c_proj.weight.normal_(std=.02);b.mlp.c_proj.weight.normal_(std=.02)
    return m


@pytest.mark.parametrize('shape',[(2,2,17,17,32),(2,4,3,7,32),(2,2,1,9,64),(1,4,31,31,128)])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
def test_flash_forward_backward_oracle(shape,dtype):
    seed_all(7);b,h,t,s,d=shape
    q=torch.randn(b,h,t,d,device='cuda',dtype=dtype,requires_grad=True)
    k=torch.randn(b,h,s,d,device='cuda',dtype=dtype,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True);go=torch.randn_like(q)
    out=history_read(q,k,v);grad=torch.autograd.grad(out,(q,k,v),go)
    args=[x.detach().double().requires_grad_() for x in (q,k,v)]
    ref=history_read(*args,backend='torch');expected=torch.autograd.grad(ref,args,go.double())
    for a,r in zip((out,*grad),(ref,*expected)):
        assert torch.isfinite(a).all() and relative(a,r.to(dtype))<(.009 if dtype==torch.bfloat16 else .002)


@pytest.mark.parametrize('query',['embedding','previous_top'])
def test_strict_past_mix_and_gradients_against_full_mask(query):
    m=make(query=query);ref=make(query=query);ref.load_state_dict(m.state_dict())
    seed_all(10)
    e=torch.randn(2,17,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    top=torch.randn_like(e,requires_grad=True);er=e.detach().clone().requires_grad_();tr=top.detach().clone().requires_grad_()
    actual=m.mix(e,top)
    h=F.rms_norm(tr.float(),(128,),eps=1e-5).to(tr.dtype);w=ref.history_projections
    q=apply_rotary_emb(w['q'](h if query=='previous_top' else er).view(2,17,2,64),ref.cos[:,:17],ref.sin[:,:17]).transpose(1,2)
    k=apply_rotary_emb(w['k'](h).view(2,17,2,64),ref.cos[:,:17],ref.sin[:,:17]).transpose(1,2)
    v=w['v'](h).view(2,17,2,64).transpose(1,2)
    mask=torch.ones(17,17,device='cuda',dtype=torch.bool).tril(-1)
    # Explicitly handle empty row zero without ever taking softmax(-inf,...).
    scores=(q.float()@k.float().transpose(-1,-2))*64**-.5
    scores=scores.masked_fill(~mask,-float('inf'));scores=torch.cat((torch.zeros_like(scores[:,:,:1]),scores[:,:,1:]),2)
    probabilities=scores.softmax(-1)*mask
    read=(probabilities@v.float()).to(e.dtype).transpose(1,2).flatten(2)
    expected=er+w['o'](read)
    go=torch.randn_like(e);actual.backward(go);expected.backward(go)
    for a,r in ((actual,expected),(e.grad,er.grad),(top.grad,tr.grad)):
        assert relative(a,r)<.025
    for p,r in zip(m.feedback_parameters(),ref.feedback_parameters()):assert relative(p.grad,r.grad)<.025
    assert torch.equal(actual[:,0],e[:,0])
    if query=='embedding':assert torch.count_nonzero(top.grad[:,-1])==0
    else:assert top.grad[:,-1].norm()>0  # Last state participates as Q, never KV.
    m.zero_grad(set_to_none=True);e.grad=None;top.grad=None
    m.mix(e,top)[:,5].float().sum().backward()
    assert torch.count_nonzero(top.grad[:,6:])==0
    if query=='embedding':assert torch.count_nonzero(top.grad[:,5])==0
    else:assert top.grad[:,5].norm()>0
    assert all(top.grad[:,j].norm()>0 for j in range(5))


@pytest.mark.parametrize('detach',[True,False])
@pytest.mark.parametrize('query',['embedding','previous_top'])
@pytest.mark.parametrize('compiled,checkpoint,passes',[(False,False,2),(True,False,3),(True,True,2)])
def test_complete_model_reference_gradients(detach,compiled,checkpoint,passes,query):
    torch._dynamo.config.recompile_limit=max(32,torch._dynamo.config.recompile_limit)
    m=make(compiled=compiled,detach=detach,checkpoint=checkpoint,query=query)
    ref=make(backend='torch',compiled=compiled,detach=detach,checkpoint=checkpoint,query=query);ref.load_state_dict(m.state_dict())
    x,y=inputs();a=m(x,y,passes=passes);b=ref(x,y,passes=passes)
    a.backward();b.backward();torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-4)
    for (n,p),(_,r) in zip(m.named_parameters(),ref.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(),n
        assert relative(p.grad,r.grad)<.04,(n,relative(p.grad,r.grad))
    assert all(p.grad.norm()>0 for p in m.feedback_parameters())


@pytest.mark.parametrize('detach',[True,False])
@pytest.mark.parametrize('query',['embedding','previous_top'])
def test_backward_policy_and_single_head(detach,query):
    m=make(detach=detach,query=query);x,y=inputs();seen=[];tops=[];fn=m.backbone
    def observe(*args,**kwargs):
        value=fn(*args,**kwargs);seen.append(torch.is_grad_enabled())
        if value.requires_grad:value.retain_grad()
        tops.append(value);return value
    with patch.object(m,'backbone',side_effect=observe),patch.object(m,'readout',wraps=m.readout) as head:
        m(x,y,passes=3).backward();assert head.call_count==1
    assert seen==[not detach,not detach,True]
    for top in tops[:-1]:
        if detach:assert not top.requires_grad
        else:
            assert torch.isfinite(top.grad).all() and top.grad.norm()>0
            if query=='embedding':assert torch.count_nonzero(top.grad[:,-1])==0
            else:assert top.grad[:,-1].norm()>0


@pytest.mark.parametrize('passes',[1,2,4])
@pytest.mark.parametrize('query',['embedding','previous_top'])
@torch.no_grad()
def test_causal_prefix_and_batch_isolation(passes,query):
    m=make(compiled=True,query=query).eval();x,_=inputs();other=x.clone();other[:,7:]=(other[:,7:]+37)%256
    assert torch.equal(m(x,passes=passes)[:,:7],m(other,passes=passes)[:,:7])
    other=x.clone();other[0]=(other[0]+21)%256
    assert torch.equal(m(x,passes=passes)[1],m(other,passes=passes)[1])


@pytest.mark.parametrize('passes',[1,2,4])
@pytest.mark.parametrize('query',['embedding','previous_top'])
@torch.no_grad()
def test_same_pass_parallel_chunked_and_token_cached(passes,query):
    m=make(query=query).eval();x,_=inputs(8);expected=m(x,passes=passes)
    cache=m.make_kv_cache(2,passes);whole=m(x,kv_cache=cache)
    cache.reset();chunks=torch.cat((m(x[:,:1],kv_cache=cache),m(x[:,1:4],kv_cache=cache),m(x[:,4:],kv_cache=cache)),1)
    cache.reset();tokens=torch.cat([m(part,kv_cache=cache) for part in x.split(1,1)],1)
    for value in (whole,chunks,tokens):assert relative(value,expected)<.02
    assert cache.get_pos()==8 and all(c.get_pos()==8 for c in cache.backbones)


@pytest.mark.parametrize('query',['embedding','previous_top'])
@torch.no_grad()
def test_cache_broadcast_and_invalid_inputs(query):
    m=make(query=query).eval();x,_=inputs(8);one=m.make_kv_cache(1,2);m(x[:1,:4],kv_cache=one)
    many=m.make_kv_cache(2,2);many.prefill(one)
    actual=m(x[:1,4:].expand(2,-1),kv_cache=many);expected=m(x[:1,4:],kv_cache=one)
    assert relative(actual,expected.expand_as(actual))<.02
    with pytest.raises(ValueError,match='count'):m(x,kv_cache=one,passes=4)
    with pytest.raises(ValueError):m.make_kv_cache(1,4).prefill(one)
    with pytest.raises(ValueError,match='Context'):m(x.repeat(1,5),kv_cache=m.make_kv_cache(2,2))
    with torch.enable_grad(),pytest.raises(ValueError,match='no_grad'):m(x,kv_cache=m.make_kv_cache(2,2))


@pytest.mark.parametrize('query',['embedding','previous_top'])
def test_baseline_initialization_p1_and_t1_identity(query):
    m=make(perturb=False,query=query);seed_all(42)
    c=replace(m.config,latent_feedback=False,feedback_source='top',feedback_attention_backend='triton',feedback_query='embedding')
    base=build_connection_model(c).prepare();x,y=inputs()
    assert all(torch.equal(v,m.state_dict()[n]) for n,v in base.state_dict().items())
    m(x,y,dense=True,passes=1).backward();base(x,y,dense=True).backward()
    for n,p in base.named_parameters():assert torch.equal(p.grad,dict(m.named_parameters())[n].grad)
    assert all(p.grad is None for p in m.feedback_parameters())
    with torch.no_grad():assert torch.equal(m(x[:,:1],passes=4),base(x[:,:1]))


@pytest.mark.parametrize('query',['embedding','previous_top'])
def test_p1_keeps_feedback_optimizer_state(query):
    m=make(query=query);opt,_=optimizer_for(m);x,y=inputs()
    m(x,y,passes=2).backward();opt.step();opt.zero_grad(set_to_none=True)
    params=list(m.feedback_parameters());weights=[p.clone() for p in params];states=[copy.deepcopy(opt.state[p]) for p in params]
    m(x,y,passes=1).backward();opt.step()
    for p,w,s in zip(params,weights,states):assert p.grad is None and torch.equal(p,w);equal_tree(s,opt.state[p])


@pytest.mark.parametrize('compiled',[False,True])
@pytest.mark.parametrize('query',['embedding','previous_top'])
def test_cce_dense_gradients(compiled,query):
    m=make(compiled=compiled,query=query);ref=make(compiled=compiled,query=query);ref.load_state_dict(m.state_dict());x,y=inputs()
    a=m(x,y,passes=2);b=ref(x,y,dense=True,passes=2)
    a.backward();b.backward();torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-4)
    for (n,p),(_,r) in zip(m.named_parameters(),ref.named_parameters()):assert relative(p.grad,r.grad)<.025,n


@pytest.mark.parametrize('query',['embedding','previous_top'])
def test_config_and_cost(query):
    c=make(query=query).config;assert ConnectionConfig.from_checkpoint(asdict(c))==c
    legacy=asdict(c);legacy.pop('feedback_query');assert ConnectionConfig.from_checkpoint(legacy).feedback_query=='embedding'
    with pytest.raises(ValueError):replace(c,feedback_bank='residual_update')
    with pytest.raises(ValueError):replace(c,feedback_attention_backend='triton')
    p1=model_cost(c,feedback_passes=1);p2=model_cost(c,feedback_passes=2)
    assert p1['additional_parameters']==4*c.width*c.width
    assert model_cost(c)['estimated_training_flops']==pytest.approx(.15*p1['estimated_training_flops']+.85*p2['estimated_training_flops'])
    assert model_cost(replace(c,feedback_detach=False),feedback_passes=2)['estimated_training_flops']>p2['estimated_training_flops']


@pytest.mark.parametrize('query',['embedding','previous_top'])
@torch.no_grad()
def test_current_position_is_query_only(query):
    m=make(query=query);seed_all(10)
    e=torch.randn(2,17,128,device='cuda',dtype=torch.bfloat16);top=torch.randn_like(e)
    changed=top.clone();changed[:,6]=torch.randn_like(changed[:,6])
    a=m.mix(e,top);b=m.mix(e,changed)
    assert torch.equal(a[:,:6],b[:,:6])
    if query=='previous_top':assert not torch.equal(a[:,6],b[:,6])
    else:assert torch.equal(a[:,6],b[:,6])
