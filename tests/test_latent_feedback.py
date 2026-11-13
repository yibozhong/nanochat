"""GPU checks for additive feedback, conditional updates and cache recurrence."""
import copy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for
from nanochat.gpt import norm
from nanochat.latent_feedback import FeedbackKVCache, sampled_passes
from nanochat.packed_training import seed_all, snapshot, restore_rng
from test_connection_training_resume import equal_tree


@pytest.fixture(autouse=True)
def gpu():
    assert torch.cuda.is_available(), 'Run via the GPU debug job'
    configure_attention('fa2')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)


def make(feedback=True, compiled=False):
    seed_all(42)
    c=ConnectionConfig(depth=2,width=128,heads=1,sequence_length=32,vocab_size=256,
        streams=1,latent_feedback=feedback)
    m=build_connection_model(c).prepare(compile_blocks=compiled)
    # Nontrivial attention/MLP, so causality and recurrence tests exercise layers.
    with torch.no_grad():
        for b in m.blocks:
            b.attn.c_proj.weight.normal_(std=.02)
            b.mlp.c_proj.weight.normal_(std=.02)
    return m


def inputs(n=16):
    x=torch.randint(0,256,(2,n),device='cuda')
    return x,(x+3)%256


def test_baseline_init_and_single_pass_identity():
    # Test untouched factory initialization, before test-specific perturbations.
    c=ConnectionConfig(depth=2,width=128,heads=1,sequence_length=32,vocab_size=256)
    seed_all(42); base=build_connection_model(c).prepare()
    seed_all(42); feed=build_connection_model(replace(c,latent_feedback=True)).prepare()
    for name,p in base.state_dict().items():
        assert torch.equal(p,feed.state_dict()[name]),name
    x,y=inputs()
    assert torch.equal(base(x),feed(x,passes=1))
    for model in (base,feed):
        (model(x,y,dense=True,passes=1) if model is feed else model(x,y,dense=True)).backward()
    for name,p in base.named_parameters():
        torch.testing.assert_close(p.grad,dict(feed.named_parameters())[name].grad,rtol=0,atol=0)
    assert feed.fusion.weight.grad is None


@pytest.mark.parametrize('passes',[1,2,4])
def test_causal_prefix(passes):
    m=make().eval();x,_=inputs()
    other=x.clone();other[:,7:]=(other[:,7:]+37)%256
    with torch.no_grad():
        a=m(x,passes=passes);b=m(other,passes=passes)
    assert torch.equal(a[:,:7],b[:,:7])
    assert not torch.equal(a[:,7:],b[:,7:])


def test_no_grad_pass_and_explicit_reference_gradient():
    m=make();ref=copy.deepcopy(m);x,y=inputs();seen=[]
    # ConnectionModel calls block.forward directly (also the compile target),
    # so Module forward hooks do not run. Observe the actual backbone call.
    backbone=m.backbone
    def observed_backbone(*args,**kwargs):
        out=backbone(*args,**kwargs)
        seen.append((torch.is_grad_enabled(),out.requires_grad))
        return out
    with patch.object(m,'backbone',side_effect=observed_backbone), patch.object(m,'readout',wraps=m.readout) as readout:
        loss=m(x,y,dense=True,passes=2)
        assert readout.call_count==1
    loss.backward()
    assert seen==[(False,False),(True,True)]
    e=norm(ref.wte(x))
    with torch.no_grad():top=ref.backbone(e)
    normalized=F.rms_norm(top.detach(),(ref.config.width,),eps=1e-5)
    projected=F.linear(normalized,ref.fusion.weight.to(e.dtype))
    mixed=torch.cat((e[:,:1],e[:,1:]+projected[:,:-1]),1)
    expected=ref.readout(norm(ref.backbone(mixed)),y,True)
    expected.backward()
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    for (name,p),(_,q) in zip(m.named_parameters(),ref.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    assert m.fusion.weight.grad.norm()>0 and m.wte.weight.grad.norm()>0


@pytest.mark.parametrize('compiled',[False,True])
def test_cce_dense_loss_and_gradient(compiled):
    m=make(compiled=compiled);x,y=inputs()
    dense=m(x,y,dense=True,passes=2);dense.backward()
    expected={n:p.grad.clone() for n,p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    cce=m(x,y,passes=2);cce.backward()
    torch.testing.assert_close(cce,dense,rtol=1e-4,atol=1e-4)
    for name,p in m.named_parameters():
        error=(p.grad.float()-expected[name].float()).norm()/expected[name].float().norm().clamp_min(1e-10)
        assert error<.025,(name,error.item())


def cache(batch=2):
    return FeedbackKVCache(batch,1,32,128,2,'cuda',torch.bfloat16)


@torch.no_grad()
def test_exact_recurrence_prefill_reset_and_broadcast():
    m=make().eval();x,_=inputs(8)
    sequential=cache();expected=m(x,kv_cache=sequential)
    chunks=cache()
    actual=torch.cat([m(x[:,:3],kv_cache=chunks),m(x[:,3:5],kv_cache=chunks),m(x[:,5:],kv_cache=chunks)],1)
    assert torch.equal(actual,expected)
    # T parallel passes reach the exact causal fixed point for T tokens.
    parallel=m(x,passes=x.shape[1])
    error=(parallel-expected).norm()/expected.norm()
    assert error<.02,error.item()
    chunks.reset()
    assert chunks.feedback_state is None and chunks.get_pos()==0
    assert torch.equal(m(x,kv_cache=chunks),expected)
    one=cache(1);m(x[:1,:4],kv_cache=one)
    many=cache(2);many.prefill(one)
    assert torch.equal(many.feedback_state[0],many.feedback_state[1])
    result=m(x[:1,4:5].expand(2,-1),kv_cache=many)
    direct=m(x[:1,4:5],kv_cache=one)
    torch.testing.assert_close(result,direct.expand(2,-1,-1),rtol=.02,atol=2e-4)


def test_single_pass_skips_fusion_momentum_and_decay():
    m=make();opt,_=optimizer_for(m);x,y=inputs()
    m(x,y,passes=2).backward();opt.step();opt.zero_grad(set_to_none=True)
    old=m.fusion.weight.clone();state=copy.deepcopy(opt.state[m.fusion.weight])
    m(x,y,passes=1).backward()
    assert m.fusion.weight.grad is None
    opt.step()
    assert torch.equal(old,m.fusion.weight)
    equal_tree(state,opt.state[m.fusion.weight])


def test_resume_preserves_conditional_optimizer_and_sampler(tmp_path):
    m=make();opt,_=optimizer_for(m);x,y=inputs()
    m(x,y,passes=2).backward();opt.step();opt.zero_grad(set_to_none=True)
    path=tmp_path/'latest.pt';torch.save(snapshot(m,opt,1,next_data_row=2),path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    restored=make();other,_=optimizer_for(restored)
    restored.load_state_dict(state['model']);other.load_state_dict(state['optimizer'])
    for passes in (1,2):
        for model,optimizer in ((m,opt),(restored,other)):
            restore_rng(state['rng']);optimizer.zero_grad(set_to_none=True)
            model(x,y,passes=passes).backward();optimizer.step()
        equal_tree(m.state_dict(),restored.state_dict())
        equal_tree(opt.state_dict(),other.state_dict())
    assert state['step']==1 and state['next_data_row']==2
    draws=[sampled_passes(42,s) for s in range(1000)]
    assert draws==[int(np.random.default_rng([42,s]).choice(2,p=[.15,.85]))+1 for s in range(1000)]
    assert .80<draws.count(2)/len(draws)<.90


def test_eval_default_is_four_passes():
    m=make().eval();x,_=inputs()
    with torch.no_grad():assert torch.equal(m(x),m(x,passes=4))


def test_invalid_configuration():
    with pytest.raises(ValueError):ConnectionConfig(latent_feedback=True,architecture='native')
    with pytest.raises(ValueError):ConnectionConfig(latent_feedback=True,variant='mhc')
    with pytest.raises(ValueError):ConnectionConfig(feedback_eval_passes=0)
