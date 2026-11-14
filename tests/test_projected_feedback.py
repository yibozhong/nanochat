"""GPU checks for independently projected, previous-token layer feedback."""
import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from nanochat.connection_cost import model_cost
from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig,build_connection_model,optimizer_for
from nanochat.gpt import norm
from nanochat.latent_feedback import FeedbackKVCache
from nanochat.packed_training import seed_all
from nanochat.projected_feedback import normalize_bank
from nanochat.projected_layer_read import projected_read,projected_read_reference
from test_connection_training_resume import equal_tree

MODES=('embedding_qkv','per_layer_qkv')


@pytest.fixture(autouse=True)
def gpu():
    assert torch.cuda.is_available(),'Run via the GPU debug allocation'
    configure_attention('fa2');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False


def relative(a,b):
    return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize('shape',[(2,17,3,128),(2,1,8,512),(2,9,5,80),(2,9,1,128)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.float16])
@pytest.mark.parametrize('shift',[False,True])
def test_read_and_all_gradients_against_double_oracle(shape,dtype,shift):
    seed_all(34);b,t,l,d=shape
    q=torch.randn(b,t,d,device='cuda',dtype=dtype,requires_grad=True)
    kv=torch.randn(b,t,l,2*d,device='cuda',dtype=dtype,requires_grad=True)
    go=torch.randn_like(q);actual=projected_read(q,kv,shift)
    gradients=torch.autograd.grad(actual,(q,kv),go)
    rq=q.detach().double().requires_grad_();rkv=kv.detach().double().requires_grad_()
    expected=projected_read_reference(rq,rkv,shift)
    refs=torch.autograd.grad(expected,(rq,rkv),go.double())
    tol=.009 if dtype==torch.bfloat16 else .002 if dtype==torch.float16 else 3e-5
    for a,r in zip((actual,*gradients),(expected,*refs)):
        assert torch.isfinite(a).all()
        assert relative(a,r.to(dtype))<tol
    if l==1:
        assert torch.count_nonzero(gradients[0])==0
        assert torch.count_nonzero(gradients[1][...,:d])==0
    if shift:
        assert torch.count_nonzero(actual[:,0])==0
        assert torch.count_nonzero(gradients[0][:,0])==0
        assert torch.count_nonzero(gradients[1][:,-1])==0


def test_production_shape_forward_backward():
    seed_all(7)
    q=torch.randn(16,2048,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    kv=torch.randn(16,2048,8,1024,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    go=torch.randn_like(q)
    a=projected_read(q,kv);ga=torch.autograd.grad(a,(q,kv),go)
    b=projected_read_reference(q,kv);gb=torch.autograd.grad(b,(q,kv),go)
    for actual,reference in zip((a,*ga),(b,*gb)):
        assert relative(actual,reference)<.009


def test_no_cross_batch_shift_and_unused_memory_gradient():
    q=torch.randn(2,3,8,device='cuda',requires_grad=True)
    kv=torch.randn(2,3,3,16,device='cuda',requires_grad=True)
    before=projected_read(q,kv)
    with torch.no_grad():kv[0,-1].fill_(100.)
    assert torch.equal(projected_read(q,kv)[1],before[1])
    loss=projected_read(q,kv)[1].sum();loss.backward()
    assert torch.count_nonzero(kv.grad[0])==0
    assert torch.count_nonzero(kv.grad[:,-1])==0


def make(mode,backend='triton',compiled=False,perturb=True,checkpoint=False):
    seed_all(42)
    c=ConnectionConfig(depth=2,width=128,heads=1,sequence_length=32,vocab_size=256,
        streams=1,latent_feedback=True,feedback_source=mode,
        feedback_attention_backend=backend,activation_checkpoint=checkpoint)
    m=build_connection_model(c).prepare(compile_blocks=compiled)
    if perturb:
        with torch.no_grad():
            for b in m.blocks:
                b.attn.c_proj.weight.normal_(std=.02)
                b.mlp.c_proj.weight.normal_(std=.02)
    return m


def inputs(length=16):
    x=torch.randint(0,256,(2,length),device='cuda')
    return x,(x+5)%256


@pytest.mark.parametrize('mode',MODES)
def test_single_pass_baseline_identity_and_gradients(mode):
    m=make(mode,perturb=False);seed_all(42)
    base=build_connection_model(replace(m.config,latent_feedback=False,feedback_source='top')).prepare()
    assert all(torch.equal(v,m.state_dict()[n]) for n,v in base.state_dict().items())
    x,y=inputs();assert torch.equal(base(x),m(x,passes=1))
    base(x,y,dense=True).backward();m(x,y,dense=True,passes=1).backward()
    for n,p in base.named_parameters():
        torch.testing.assert_close(p.grad,dict(m.named_parameters())[n].grad,rtol=0,atol=0)
    assert all(p.grad is None for p in m.feedback_parameters())


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('compiled,checkpoint',[(False,False),(True,False),(True,True)])
def test_model_reference_gradients(mode,compiled,checkpoint):
    m=make(mode,compiled=compiled,checkpoint=checkpoint)
    ref=make(mode,backend='torch');ref.load_state_dict(m.state_dict())
    x,y=inputs();a=m(x,y,passes=2);b=ref(x,y,passes=2)
    a.backward();b.backward()
    torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-4)
    for (n,p),(_,r) in zip(m.named_parameters(),ref.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(),n
        assert relative(p.grad,r.grad)<.04,(n,relative(p.grad,r.grad))
    for reader in m.feedback_readers:
        assert all(p.grad.norm()>0 for p in reader.parameters())


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('passes',[2,4])
@torch.no_grad()
def test_causal_prefix(mode,passes):
    m=make(mode).eval();x,_=inputs();other=x.clone();other[:,7:]=(other[:,7:]+31)%256
    assert torch.equal(m(x,passes=passes)[:,:7],m(other,passes=passes)[:,:7])


@pytest.mark.parametrize('mode',MODES)
def test_detached_bank_trainable_kv_and_single_head_call(mode):
    m=make(mode);x,y=inputs();seen=[];fn=m.backbone
    def observe(*args,**kwargs):
        out=fn(*args,**kwargs)
        seen.append((torch.is_grad_enabled(),out[1].requires_grad) if isinstance(out,tuple)
            else (torch.is_grad_enabled(),out.requires_grad))
        return out
    with patch.object(m,'backbone',side_effect=observe),patch.object(m,'readout',wraps=m.readout) as head:
        m(x,y,passes=2).backward();assert head.call_count==1
    assert seen==[(False,False),(True,True)]
    m.zero_grad(set_to_none=True)
    raw=torch.randn(2,16,2,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    e=norm(m.wte(x));m._pass(e,raw).float().square().mean().backward()
    assert raw.grad is None
    for r in m.feedback_readers:
        assert r.k_proj.weight.grad.norm()>0 and r.v_proj.weight.grad.norm()>0


@torch.no_grad()
def test_per_block_queries_and_independent_projections():
    m=make('per_layer_qkv');x,_=inputs();e=norm(m.wte(x))
    _,bank=m.backbone(e,return_layers=True);observed=[]
    def hook(module,args):observed.append(args[0].clone())
    handles=[r.register_forward_pre_hook(hook) for r in m.feedback_readers]
    m._pass(e,bank)
    for h in handles:h.remove()
    assert len(observed)==m.config.depth
    assert torch.equal(observed[0],e)
    assert not torch.equal(observed[1],e)
    params=list(m.feedback_parameters())
    assert len(params)==4*m.config.depth and len({p.data_ptr() for p in params})==len(params)


@pytest.mark.parametrize('mode',MODES)
@torch.no_grad()
def test_cached_recurrence_reset_and_prefill(mode):
    def cache(b=2):return FeedbackKVCache(b,1,32,128,2,'cuda',torch.bfloat16)
    m=make(mode).eval();x,_=inputs(8)
    expected=m(x,kv_cache=cache())
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


@pytest.mark.parametrize('mode',MODES)
def test_p1_skips_conditional_parameters_and_optimizer_state(mode):
    m=make(mode);opt,_=optimizer_for(m);x,y=inputs()
    m(x,y,passes=2).backward();opt.step();opt.zero_grad(set_to_none=True)
    params=list(m.feedback_parameters());before=[p.clone() for p in params]
    states=[copy.deepcopy(opt.state[p]) for p in params]
    m(x,y,passes=1).backward();assert all(p.grad is None for p in params)
    opt.step()
    for p,old,state in zip(params,before,states):
        assert torch.equal(p,old);equal_tree(state,opt.state[p])


@pytest.mark.parametrize('mode',MODES)
def test_parameter_and_flop_accounting(mode):
    m=make(mode);c=m.config;count=4*c.width**2*(c.depth if mode=='per_layer_qkv' else 1)
    assert sum(p.numel() for p in m.feedback_parameters())==count
    cost=model_cost(c,batch=2,feedback_passes=1)
    base=model_cost(replace(c,latent_feedback=False,feedback_source='top'),batch=2)
    assert cost['additional_parameters']==count
    assert cost['estimated_training_flops']==base['estimated_training_flops']
    average=model_cost(c,batch=2)['estimated_training_flops']
    p2=model_cost(c,batch=2,feedback_passes=2)['estimated_training_flops']
    assert average==pytest.approx(.15*base['estimated_training_flops']+.85*p2)
