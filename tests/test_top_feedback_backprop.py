"""Full-gradient top-hidden feedback, retaining the original LoRM forward."""
import copy
from dataclasses import asdict,replace
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from nanochat.connection_model import ConnectionConfig,build_connection_model,optimizer_for
from nanochat.connection_cost import model_cost
from nanochat.gpt import norm
from test_latent_feedback import gpu,make as detached_make,inputs,cache
from test_projected_feedback import relative
from test_connection_training_resume import equal_tree


def make(compiled=False,checkpoint=False):
    template=detached_make()
    c=replace(template.config,feedback_detach=False,activation_checkpoint=checkpoint)
    model=build_connection_model(c).prepare(compile_blocks=compiled)
    model.load_state_dict(template.state_dict());return model


@pytest.mark.parametrize('compiled,checkpoint,passes',[(False,False,2),(True,False,2),(True,True,3)])
def test_full_gradient_manual_unroll(compiled,checkpoint,passes):
    torch._dynamo.config.recompile_limit=max(32,torch._dynamo.config.recompile_limit)
    m=make(compiled,checkpoint);ref=make(compiled,checkpoint);ref.load_state_dict(m.state_dict())
    def mix(e,top):
        fed=F.linear(F.rms_norm(top,(top.shape[-1],),eps=1e-5),ref.fusion.weight.to(e.dtype))
        return torch.cat((e[:,:1],e[:,1:]+fed[:,:-1]),1)
    fn=torch.compile(mix,fullgraph=True,dynamic=False) if compiled else mix
    x,y=inputs();loss=m(x,y,passes=passes)
    e=norm(ref.wte(x));h=ref.backbone(e)
    for _ in range(passes-1):h=ref.backbone(fn(e,h))
    expected=ref.readout(norm(h),y)
    loss.backward();expected.backward()
    torch.testing.assert_close(loss,expected,rtol=2e-4,atol=2e-4)
    for (n,p),(_,r) in zip(m.named_parameters(),ref.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(),n
        assert relative(p.grad,r.grad)<.04,(n,relative(p.grad,r.grad))


@pytest.mark.parametrize('passes',[2,4])
def test_previous_pass_gradients_and_single_head(passes):
    m=make();x,y=inputs();states=[];fn=m.backbone
    def observe(*args,**kwargs):
        top=fn(*args,**kwargs)
        assert torch.is_grad_enabled() and top.requires_grad
        top.retain_grad();states.append(top);return top
    with patch.object(m,'backbone',side_effect=observe),patch.object(m,'readout',wraps=m.readout) as head:
        m(x,y,passes=passes).backward();assert head.call_count==1
    assert len(states)==passes
    for top in states[:-1]:
        assert top.grad is not None and torch.isfinite(top.grad).all() and top.grad.norm()>0
        assert torch.count_nonzero(top.grad[:,-1])==0
    assert m.fusion.weight.grad.norm()>0


@pytest.mark.parametrize('passes',[1,2,4])
@torch.no_grad()
def test_forward_identity_and_causality(passes):
    m=make(compiled=True).eval();old=detached_make(compiled=True).eval();old.load_state_dict(m.state_dict())
    x,_=inputs();other=x.clone();other[:,7:]=(other[:,7:]+37)%256
    assert torch.equal(m(x,passes=passes),old(x,passes=passes))
    assert torch.equal(m(x,passes=passes)[:,:7],m(other,passes=passes)[:,:7])


@torch.no_grad()
def test_exact_cache_and_parallel_convergence():
    m=make().eval();x,_=inputs(8)
    expected=m(x,kv_cache=cache());split=cache()
    actual=torch.cat((m(x[:,:3],kv_cache=split),m(x[:,3:],kv_cache=split)),1)
    assert torch.equal(actual,expected)
    assert relative(m(x,passes=8),expected)<.02
    split.reset();assert split.feedback_state is None and split.get_pos()==0
    assert torch.equal(m(x,kv_cache=split),expected)


def test_p1_preserves_fusion_optimizer():
    m=make();opt,_=optimizer_for(m);x,y=inputs()
    m(x,y,passes=2).backward();opt.step();opt.zero_grad(set_to_none=True)
    old=m.fusion.weight.clone();state=copy.deepcopy(opt.state[m.fusion.weight])
    m(x,y,passes=1).backward();assert m.fusion.weight.grad is None
    opt.step();assert torch.equal(m.fusion.weight,old);equal_tree(state,opt.state[m.fusion.weight])


def test_p1_gradients_unchanged():
    m=make();old=detached_make();old.load_state_dict(m.state_dict());x,y=inputs()
    m(x,y,dense=True,passes=1).backward();old(x,y,dense=True,passes=1).backward()
    for p,r in zip(m.parameters(),old.parameters()):
        if p.grad is None:assert r.grad is None
        else:assert torch.equal(p.grad,r.grad)


def test_config_and_compute_accounting():
    c=make().config
    assert ConnectionConfig.from_checkpoint(asdict(c))==c
    legacy=asdict(c);legacy.pop('feedback_detach');assert ConnectionConfig.from_checkpoint(legacy).feedback_detach
    with pytest.raises(ValueError):replace(c,feedback_source='layers')
    with pytest.raises(ValueError):replace(c,latent_feedback=False)
    d=replace(c,feedback_detach=True)
    p1=model_cost(c,feedback_passes=1);p2=model_cost(c,feedback_passes=2)
    assert p1['estimated_training_flops']==model_cost(d,feedback_passes=1)['estimated_training_flops']
    assert p2['estimated_training_flops']>model_cost(d,feedback_passes=2)['estimated_training_flops']
    assert model_cost(c)['estimated_training_flops']==pytest.approx(.15*p1['estimated_training_flops']+.85*p2['estimated_training_flops'])
    assert p2['additional_parameters']==c.width**2


@pytest.mark.parametrize('compiled',[False,True])
def test_cce_matches_dense_with_full_gradients(compiled):
    m=make(compiled);x,y=inputs()
    a=m(x,y,dense=True,passes=2);a.backward()
    grads={n:p.grad.clone() for n,p in m.named_parameters()}
    m.zero_grad(set_to_none=True);b=m(x,y,passes=2);b.backward()
    torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-4)
    for n,p in m.named_parameters():assert relative(p.grad,grads[n])<.025,n
