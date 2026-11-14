"""GPU oracle, causal/cache, detached-gradient and optimizer checks."""
import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig,build_connection_model,optimizer_for
from nanochat.layer_feedback import layer_read,layer_read_reference
from nanochat.latent_feedback import FeedbackKVCache
from nanochat.packed_training import seed_all
from test_connection_training_resume import equal_tree


@pytest.fixture(autouse=True)
def gpu():
    assert torch.cuda.is_available(),'Run in the GPU allocation'
    configure_attention('fa2');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False


def relative(a,b):
    return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize('shape',[(2,17,3,128),(2,1,8,512),(2,9,5,80),(2,9,1,128)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('shift',[False,True])
def test_fused_read_against_double_oracle(shape,dtype,shift):
    seed_all(34);b,t,l,d=shape
    q=torch.randn(b,t,d,device='cuda',dtype=dtype,requires_grad=True)
    bank=torch.randn(b,t,l,d,device='cuda',dtype=dtype)
    grad=torch.randn_like(q)
    out=layer_read(q,bank,shift);dq=torch.autograd.grad(out,q,grad)[0]
    refq=q.detach().double().requires_grad_()
    expected=layer_read_reference(refq,bank.double(),shift)
    expected_dq=torch.autograd.grad(expected,refq,grad.double())[0].to(dtype)
    assert torch.isfinite(out).all() and torch.isfinite(dq).all()
    tol=.009 if dtype==torch.bfloat16 else 3e-5
    assert relative(out,expected.to(dtype))<tol
    if l==1:
        assert torch.count_nonzero(dq)==0
    else:
        assert relative(dq,expected_dq)<tol
    if shift:
        assert torch.count_nonzero(out[:,0])==0 and torch.count_nonzero(dq[:,0])==0


def test_production_shape_fused_forward_backward():
    seed_all(7)
    q=torch.randn(16,2048,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    bank=torch.randn(16,2048,8,512,device='cuda',dtype=torch.bfloat16)
    grad=torch.randn_like(q)
    actual=layer_read(q,bank);dq=torch.autograd.grad(actual,q,grad)[0]
    refq=q.detach().clone().requires_grad_()
    expected=layer_read_reference(refq,bank)
    reference_grad=torch.autograd.grad(expected,refq,grad)[0]
    assert relative(actual,expected)<.009
    assert relative(dq,reference_grad)<.009


def test_query_selects_layers_and_no_cross_batch_shift():
    bank=torch.eye(4,device='cuda')[:3].view(1,1,3,4)
    q=torch.tensor([[[30.,0.,0.,0.]]],device='cuda')
    first=layer_read(q,bank,shift=False)
    third=layer_read(q.roll(2,-1),bank,shift=False)
    assert first[0,0,0]>1.99 and third[0,0,2]>1.99
    q=torch.randn(2,3,4,device='cuda')
    bank=torch.randn(2,3,3,4,device='cuda')
    old=layer_read(q,bank)
    bank[0,-1].fill_(100.)
    assert torch.equal(layer_read(q,bank)[1],old[1])
    with pytest.raises(ValueError,match='detached'):
        layer_read(q,bank.requires_grad_())


def make(backend='triton',compiled=False,perturb=True,checkpoint=False):
    seed_all(42)
    c=ConnectionConfig(depth=2,width=128,heads=1,sequence_length=32,vocab_size=256,streams=1,
        latent_feedback=True,feedback_source='layers',feedback_attention_backend=backend,
        activation_checkpoint=checkpoint)
    m=build_connection_model(c).prepare(compile_blocks=compiled)
    if perturb:
        with torch.no_grad():
            for block in m.blocks:
                block.attn.c_proj.weight.normal_(std=.02)
                block.mlp.c_proj.weight.normal_(std=.02)
    return m


def inputs(length=16):
    x=torch.randint(0,256,(2,length),device='cuda')
    return x,(x+5)%256


def test_single_pass_is_identical_to_baseline():
    m=make(perturb=False)
    seed_all(42)
    base=build_connection_model(replace(m.config,latent_feedback=False,feedback_source='top')).prepare()
    assert all(torch.equal(v,m.state_dict()[n]) for n,v in base.state_dict().items())
    x,y=inputs()
    assert torch.equal(base(x),m(x,passes=1))
    base(x,y,dense=True).backward();m(x,y,dense=True,passes=1).backward()
    for n,p in base.named_parameters():
        torch.testing.assert_close(p.grad,dict(m.named_parameters())[n].grad,rtol=0,atol=0)
    assert m.feedback_query.weight.grad is None and m.fusion.weight.grad is None


@pytest.mark.parametrize('compiled,checkpoint',[(False,False),(True,False),(True,True)])
def test_model_matches_reference_gradients(compiled,checkpoint):
    m=make(compiled=compiled,checkpoint=checkpoint)
    ref=make(backend='torch');ref.load_state_dict(m.state_dict())
    x,y=inputs();a=m(x,y,passes=2);b=ref(x,y,passes=2)
    a.backward();b.backward()
    torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-4)
    for (n,p),(_,r) in zip(m.named_parameters(),ref.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(),n
        assert relative(p.grad,r.grad)<.04,(n,relative(p.grad,r.grad))
    assert m.feedback_query.weight.grad.norm()>0 and m.fusion.weight.grad.norm()>0


@pytest.mark.parametrize('passes',[2,4])
@torch.no_grad()
def test_future_tokens_cannot_change_prefix(passes):
    m=make().eval();x,_=inputs();other=x.clone();other[:,7:]=(other[:,7:]+31)%256
    assert torch.equal(m(x,passes=passes)[:,:7],m(other,passes=passes)[:,:7])


def test_preliminary_bank_has_no_graph_and_head_runs_once():
    m=make();x,y=inputs();seen=[];fn=m.backbone
    def observe(*args,**kwargs):
        out=fn(*args,**kwargs)
        if isinstance(out,tuple):
            seen.append((torch.is_grad_enabled(),out[0].requires_grad,out[1].requires_grad))
        else:seen.append((torch.is_grad_enabled(),out.requires_grad))
        return out
    with patch.object(m,'backbone',side_effect=observe),patch.object(m,'readout',wraps=m.readout) as readout:
        m(x,y,passes=2).backward()
        assert readout.call_count==1
    assert seen==[(False,False,False),(True,True)]


@torch.no_grad()
def test_cached_recurrence_reset_and_branch_prefill():
    def cache(b=2):return FeedbackKVCache(b,1,32,128,2,'cuda',torch.bfloat16)
    m=make().eval();x,_=inputs(8)
    full=cache();expected=m(x,kv_cache=full)
    split=cache();actual=torch.cat((m(x[:,:3],kv_cache=split),m(x[:,3:],kv_cache=split)),1)
    assert torch.equal(actual,expected)
    assert split.feedback_state.shape==(2,1,2,128)
    assert relative(m(x,passes=8),expected)<.02
    split.reset();assert split.feedback_state is None and split.get_pos()==0
    assert torch.equal(m(x,kv_cache=split),expected)
    one=cache(1);m(x[:1,:4],kv_cache=one)
    many=cache(2);many.prefill(one)
    assert torch.equal(many.feedback_state[0],many.feedback_state[1])
    expected=m(x[:1,4:5],kv_cache=one)
    actual=m(x[:1,4:5].expand(2,-1),kv_cache=many)
    assert relative(actual,expected.expand_as(actual))<.02


def test_single_pass_preserves_both_conditional_parameters_and_state():
    m=make();opt,_=optimizer_for(m);x,y=inputs()
    m(x,y,passes=2).backward();opt.step();opt.zero_grad(set_to_none=True)
    before={n:p.clone() for n,p in m.named_parameters() if n.startswith(('fusion.','feedback_query.'))}
    state=copy.deepcopy(opt.state[m.fusion.weight])
    m(x,y,passes=1).backward()
    assert m.feedback_query.weight.grad is None and m.fusion.weight.grad is None
    opt.step()
    for n,p in m.named_parameters():
        if n in before:assert torch.equal(p,before[n])
    equal_tree(state,opt.state[m.fusion.weight])
