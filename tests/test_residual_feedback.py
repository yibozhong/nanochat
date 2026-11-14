"""Detached attention over previous-token block residual updates."""
import copy
from dataclasses import asdict,replace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from nanochat.connection_model import ConnectionConfig,build_connection_model,optimizer_for
from nanochat.connection_cost import model_cost
from nanochat.gpt import norm
from nanochat.latent_feedback import FeedbackKVCache
from nanochat.packed_training import seed_all
from test_projected_feedback import gpu,relative,inputs,make as hidden_make
from test_connection_training_resume import equal_tree


def make(backend='triton',compiled=False,checkpoint=False,perturb=True):
    template=hidden_make('per_layer_qkv',perturb=perturb)
    c=replace(template.config,feedback_bank='residual_update',feedback_attention_backend=backend,
        activation_checkpoint=checkpoint)
    m=build_connection_model(c).prepare(compile_blocks=compiled)
    m.load_state_dict(template.state_dict());return m


@pytest.mark.parametrize('with_feedback',[False,True])
@torch.no_grad()
def test_bank_is_branch_sum_excluding_identity_and_feedback(with_feedback):
    m=make();x,_=inputs();e=norm(m.wte(x));parts=[];handles=[]
    for block in m.blocks:
        handles.append(block.attn.register_forward_hook(lambda module,args,out:parts.append(out.clone())))
        handles.append(block.mlp.register_forward_hook(lambda module,args,out:parts.append(out.clone())))
    previous=torch.randn(2,16,2,128,device='cuda',dtype=e.dtype) if with_feedback else None
    top,bank=m._pass(e,previous,collect=True)
    for handle in handles:handle.remove()
    expected=torch.stack([(parts[2*i].float()+parts[2*i+1].float()).to(e.dtype) for i in range(2)],2)
    assert torch.equal(bank,expected)
    assert torch.equal(top,m._pass(e,previous))


class ConstantBranch(nn.Module):
    def forward(self,x,*args,**kwargs):return torch.full_like(x,1/1024)


@pytest.mark.parametrize('compiled',[False,True])
@torch.no_grad()
def test_small_updates_survive_bf16_cancellation(compiled):
    m=make();b=m.blocks[0];b.attn=ConstantBranch();b.mlp=ConstantBranch()
    fn=torch.compile(b.forward_with_update,fullgraph=True) if compiled else b.forward_with_update
    x=torch.ones(2,4,128,device='cuda',dtype=torch.bfloat16)
    top,update=fn(x,m.cos[:,:4],m.sin[:,:4])
    assert torch.equal(top,x)
    assert torch.equal(update,torch.full_like(x,1/512))


def test_zero_initial_bank_and_single_pass_identity():
    m=make(perturb=False);x,y=inputs();seed_all(42)
    base=build_connection_model(replace(m.config,latent_feedback=False,feedback_source='top',feedback_bank='hidden')).prepare()
    assert all(torch.equal(v,m.state_dict()[n]) for n,v in base.state_dict().items())
    with torch.no_grad():
        e=norm(m.wte(x));_,bank=m._pass(e,collect=True)
        assert torch.count_nonzero(bank)==0
        assert torch.equal(m(x,passes=2),base(x))
        previous=torch.randn_like(bank);_,again=m._pass(e,previous,collect=True)
        assert torch.count_nonzero(again)==0  # Feedback itself is not an update.
    m(x,y,dense=True,passes=1).backward();base(x,y,dense=True).backward()
    for n,p in base.named_parameters():assert torch.equal(p.grad,dict(m.named_parameters())[n].grad)
    assert all(p.grad is None for p in m.feedback_parameters())


@pytest.mark.parametrize('compiled,checkpoint',[(False,False),(True,False),(True,True)])
def test_reference_gradients(compiled,checkpoint):
    torch._dynamo.config.recompile_limit=max(32,torch._dynamo.config.recompile_limit)
    m=make(compiled=compiled,checkpoint=checkpoint)
    ref=make(backend='torch',compiled=compiled,checkpoint=checkpoint);ref.load_state_dict(m.state_dict())
    x,y=inputs();a=m(x,y,passes=3);b=ref(x,y,passes=3)
    a.backward();b.backward();torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-4)
    for (n,p),(_,r) in zip(m.named_parameters(),ref.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(),n
        assert relative(p.grad,r.grad)<.04,(n,relative(p.grad,r.grad))
    assert all(p.grad.norm()>0 for p in m.feedback_parameters())


@pytest.mark.parametrize('passes',[2,4])
@torch.no_grad()
def test_causal_prefix(passes):
    m=make(compiled=True).eval();x,_=inputs();other=x.clone();other[:,7:]=(other[:,7:]+31)%256
    assert torch.equal(m(x,passes=passes)[:,:7],m(other,passes=passes)[:,:7])


def test_bank_detached_head_once_and_all_readers_learn():
    m=make();x,y=inputs();observed=[];fn=m.backbone
    def observe(*args,**kwargs):
        value=fn(*args,**kwargs)
        if isinstance(value,tuple):observed.append((torch.is_grad_enabled(),value[1].requires_grad))
        return value
    with patch.object(m,'backbone',side_effect=observe),patch.object(m,'readout',wraps=m.readout) as head:
        m(x,y,passes=3).backward();assert head.call_count==1
    assert observed==[(False,False),(False,False)]
    assert all(p.grad.norm()>0 for p in m.feedback_parameters())


@torch.no_grad()
def test_cached_recurrence_and_chunking():
    def cache(b=2):return FeedbackKVCache(b,1,32,128,2,'cuda',torch.bfloat16)
    m=make().eval();x,_=inputs(8)
    entire=cache();expected=m(x,kv_cache=entire)
    split=cache();actual=torch.cat((m(x[:,:3],kv_cache=split),m(x[:,3:],kv_cache=split)),1)
    assert torch.equal(actual,expected)
    assert relative(m(x,passes=8),expected)<.02
    assert entire.feedback_state.shape==(2,1,2,128)
    split.reset();assert split.feedback_state is None and split.get_pos()==0
    assert torch.equal(m(x,kv_cache=split),expected)


def test_p1_preserves_feedback_optimizer_state():
    m=make();opt,_=optimizer_for(m);x,y=inputs()
    m(x,y,passes=2).backward();opt.step();opt.zero_grad(set_to_none=True)
    params=list(m.feedback_parameters());old=[p.clone() for p in params]
    states=[copy.deepcopy(opt.state[p]) for p in params]
    m(x,y,passes=1).backward();assert all(p.grad is None for p in params)
    opt.step()
    for p,v,s in zip(params,old,states):assert torch.equal(p,v);equal_tree(s,opt.state[p])


def test_config_legacy_roundtrip_and_cost():
    m=make();c=m.config
    assert ConnectionConfig.from_checkpoint(asdict(c))==c
    legacy=asdict(c);legacy.pop('feedback_bank');assert ConnectionConfig.from_checkpoint(legacy).feedback_bank=='hidden'
    assert not replace(c,feedback_detach=False).feedback_detach
    with pytest.raises(ValueError,match='per-layer'):replace(c,feedback_source='embedding_qkv')
    base=model_cost(replace(c,feedback_bank='hidden'),batch=2,feedback_passes=1)
    assert model_cost(c,batch=2,feedback_passes=1)['estimated_training_flops']==base['estimated_training_flops']
    hidden=model_cost(replace(c,feedback_bank='hidden'),batch=2,feedback_passes=2)
    actual=model_cost(c,batch=2,feedback_passes=2)
    assert actual['estimated_training_flops']-hidden['estimated_training_flops']==2*c.sequence_length*c.depth*c.width
    assert actual['additional_parameters']==hidden['additional_parameters']
