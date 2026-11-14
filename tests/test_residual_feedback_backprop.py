"""Full gradients through previous-token residual-update attention banks."""
import copy
from dataclasses import asdict,replace
from unittest.mock import patch

import pytest
import torch

from nanochat.connection_model import ConnectionConfig,build_connection_model,optimizer_for
from nanochat.connection_cost import model_cost
from nanochat.latent_feedback import FeedbackKVCache
from test_residual_feedback import make as detached_make
from test_projected_feedback import gpu,relative,inputs
from test_connection_training_resume import equal_tree


def make(backend='triton',compiled=False,checkpoint=False,perturb=True):
    template=detached_make(perturb=perturb)
    c=replace(template.config,feedback_detach=False,feedback_attention_backend=backend,
        activation_checkpoint=checkpoint)
    m=build_connection_model(c).prepare(compile_blocks=compiled)
    m.load_state_dict(template.state_dict());return m


@pytest.mark.parametrize('compiled,checkpoint,passes',[(False,False,2),(True,False,3),(True,True,2)])
def test_complete_model_gradients(compiled,checkpoint,passes):
    torch._dynamo.config.recompile_limit=max(32,torch._dynamo.config.recompile_limit)
    m=make(compiled=compiled,checkpoint=checkpoint)
    ref=make(backend='torch',compiled=compiled,checkpoint=checkpoint);ref.load_state_dict(m.state_dict())
    x,y=inputs();a=m(x,y,passes=passes);b=ref(x,y,passes=passes)
    a.backward();b.backward();torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-4)
    for (n,p),(_,r) in zip(m.named_parameters(),ref.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(),n
        assert relative(p.grad,r.grad)<.04,(n,relative(p.grad,r.grad))
    assert all(p.grad.norm()>0 for p in m.feedback_parameters())


@pytest.mark.parametrize('passes',[2,4])
def test_every_prior_bank_receives_gradients_and_head_runs_once(passes):
    m=make();x,y=inputs();banks=[];fn=m.backbone
    def observe(*args,**kwargs):
        value=fn(*args,**kwargs)
        if isinstance(value,tuple):
            bank=value[1];assert torch.is_grad_enabled() and bank.requires_grad
            bank.retain_grad();banks.append(bank)
        return value
    with patch.object(m,'backbone',side_effect=observe),patch.object(m,'readout',wraps=m.readout) as head:
        m(x,y,passes=passes).backward();assert head.call_count==1
    assert len(banks)==passes-1
    for bank in banks:
        assert bank.grad is not None and torch.isfinite(bank.grad).all()
        assert all(bank.grad[:,:,i].norm()>0 for i in range(m.config.depth))
        assert torch.count_nonzero(bank.grad[:,-1])==0


@pytest.mark.parametrize('passes',[1,2,4])
@torch.no_grad()
def test_forward_unchanged_and_causal(passes):
    m=make(compiled=True).eval();old=detached_make(compiled=True).eval();old.load_state_dict(m.state_dict())
    x,_=inputs();other=x.clone();other[:,7:]=(other[:,7:]+31)%256
    assert torch.equal(m(x,passes=passes),old(x,passes=passes))
    assert torch.equal(m(x,passes=passes)[:,:7],m(other,passes=passes)[:,:7])


@pytest.mark.parametrize('compiled',[False,True])
def test_zero_bank_has_finite_connected_gradient(compiled):
    m=make(compiled=compiled,perturb=False)
    ref=make(backend='torch',compiled=compiled,perturb=False);ref.load_state_dict(m.state_dict())
    old=detached_make(compiled=compiled,perturb=False);old.load_state_dict(m.state_dict())
    x,y=inputs()
    with torch.no_grad():assert torch.equal(m(x,passes=2),m(x,passes=1))
    for model in (m,ref,old):model(x,y,passes=2).backward()
    for (n,p),(_,r) in zip(m.named_parameters(),ref.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(),n
        assert relative(p.grad,r.grad)<.04,(n,relative(p.grad,r.grad))
    # A zero forward bank still carries a nonzero derivative to its producers.
    assert any(not torch.equal(a.attn.c_proj.weight.grad,b.attn.c_proj.weight.grad)
        for a,b in zip(m.blocks,old.blocks))


@pytest.mark.parametrize('compiled',[False,True])
def test_collected_update_and_hidden_joint_backward(compiled):
    m=make();ref=make();ref.load_state_dict(m.state_dict())
    b=m.blocks[0];rb=ref.blocks[0]
    def manual(x,cos,sin):
        attention=rb.attention_branch(x,cos,sin)
        middle=x+attention;mlp=rb.mlp_branch(middle)
        return middle+mlp,(attention.float()+mlp.float()).to(x.dtype)
    actual=torch.compile(b.forward_with_update,fullgraph=True) if compiled else b.forward_with_update
    expected=torch.compile(manual,fullgraph=True) if compiled else manual
    x=torch.randn(2,4,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    rx=x.detach().clone().requires_grad_();go=torch.randn_like(x);gu=torch.randn_like(x)
    a,u=actual(x,m.cos[:,:4],m.sin[:,:4]);r,v=expected(rx,ref.cos[:,:4],ref.sin[:,:4])
    ((a.float()*go).sum()+(u.float()*gu).sum()).backward()
    ((r.float()*go).sum()+(v.float()*gu).sum()).backward()
    for left,right in ((a,r),(u,v),(x.grad,rx.grad)):
        assert torch.isfinite(left).all() and relative(left,right)<.04
    for p,r in zip(b.parameters(),rb.parameters()):
        assert torch.isfinite(p.grad).all() and relative(p.grad,r.grad)<.04


@torch.no_grad()
def test_exact_cache_chunking_and_parallel_convergence():
    def cache():return FeedbackKVCache(2,1,32,128,2,'cuda',torch.bfloat16)
    m=make().eval();x,_=inputs(8);expected=m(x,kv_cache=cache());split=cache()
    actual=torch.cat((m(x[:,:3],kv_cache=split),m(x[:,3:],kv_cache=split)),1)
    assert torch.equal(actual,expected)
    assert relative(m(x,passes=8),expected)<.02
    split.reset();assert split.feedback_state is None and split.get_pos()==0
    assert torch.equal(m(x,kv_cache=split),expected)


def test_single_pass_gradients_and_feedback_optimizer_state():
    m=make();opt,_=optimizer_for(m);x,y=inputs()
    m(x,y,passes=2).backward();opt.step();opt.zero_grad(set_to_none=True)
    old=detached_make();old.load_state_dict(m.state_dict())
    params=list(m.feedback_parameters());weights=[p.clone() for p in params]
    states=[copy.deepcopy(opt.state[p]) for p in params]
    m(x,y,dense=True,passes=1).backward();old(x,y,dense=True,passes=1).backward()
    for p,r in zip(m.parameters(),old.parameters()):
        if p.grad is None:assert r.grad is None
        else:assert torch.equal(p.grad,r.grad)
    assert all(p.grad is None for p in params)
    opt.step()
    for p,w,s in zip(params,weights,states):assert torch.equal(p,w);equal_tree(s,opt.state[p])


@pytest.mark.parametrize('passes',[2,3])
def test_cce_matches_dense_full_gradient(passes):
    m=make();ref=make();ref.load_state_dict(m.state_dict());x,y=inputs()
    a=m(x,y,passes=passes);b=ref(x,y,dense=True,passes=passes)
    a.backward();b.backward();torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-4)
    for (n,p),(_,r) in zip(m.named_parameters(),ref.named_parameters()):
        assert torch.isfinite(p.grad).all() and relative(p.grad,r.grad)<.025,(n,relative(p.grad,r.grad))


def test_config_and_compute_accounting():
    c=make().config;assert ConnectionConfig.from_checkpoint(asdict(c))==c
    legacy=asdict(c);legacy.pop('feedback_detach');assert ConnectionConfig.from_checkpoint(legacy).feedback_detach
    hidden=replace(c,feedback_bank='hidden')
    p1=model_cost(c,feedback_passes=1);p2=model_cost(c,feedback_passes=2)
    assert p1['estimated_training_flops']==model_cost(hidden,feedback_passes=1)['estimated_training_flops']
    assert p2['estimated_training_flops']>model_cost(replace(c,feedback_detach=True),feedback_passes=2)['estimated_training_flops']
    assert p2['additional_parameters']==model_cost(hidden)['additional_parameters']
    assert model_cost(c)['estimated_training_flops']==pytest.approx(.15*p1['estimated_training_flops']+.85*p2['estimated_training_flops'])
