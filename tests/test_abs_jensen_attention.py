"""Absolute Jensen gap: independent oracle, kink convention, fused integration."""
from dataclasses import asdict
import pytest
import torch

from nanochat.moment_attention import value_moments
from nanochat.triton_moment_attention import fused_attention
from nanochat.connection_model import ConnectionConfig,optimizer_for
from nanochat.connection_cost import model_cost
import test_triton_moments as shared
import test_triton_resume as resume

setup=shared.setup
MODE='jensen_abs'


def oracle(q,k,v):
    outputs=[]
    for i in range(q.shape[1]):
        values=v[:,:i+1]
        logits=torch.einsum('bhd,bshd->bhs',q[:,i],k[:,:i+1])/q.shape[-1]**.5
        a=logits.softmax(-1).transpose(1,2).unsqueeze(-1)
        mean=(a*values).sum(1)
        outputs.append((a*values.abs()).sum(1)-mean.abs())
    return torch.stack(outputs,1)


@pytest.mark.parametrize('compiled',[False,True])
def test_reference_formula_and_all_qkv_gradients(compiled):
    dtype=torch.float32 if compiled else torch.float64
    xs=[torch.randn(2,16,2,8,device='cuda',dtype=dtype,requires_grad=True) for _ in range(3)]
    fn=torch.compile(value_moments,fullgraph=True,dynamic=False) if compiled else value_moments
    actual=fn(*xs,centered=False,query_chunk=4,nonlinearity='abs')
    refs=[x.detach().double().requires_grad_() for x in xs]
    expected=oracle(*refs);tol=3e-6 if compiled else 3e-12
    torch.testing.assert_close(actual.double(),expected,atol=tol,rtol=tol)
    up=torch.randn_like(actual)
    ga=torch.autograd.grad((actual*up).sum(),xs)
    ge=torch.autograd.grad((expected*up.double()).sum(),refs)
    for a,e in zip(ga,ge):assert (a.double()-e).norm()/e.norm().clamp_min(1e-12)<tol


@pytest.mark.parametrize('t,d',[(17,32),(129,64),(2048,128)])
def test_fused_forward_and_gradients(t,d):
    xs=[torch.randn(1,t,2,d,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    refs=[x.detach().clone().requires_grad_() for x in xs]
    actual=fused_attention(*xs,MODE);expected=shared.reference(*refs,MODE)
    weights=[torch.randn_like(x) for x in actual]
    grads=torch.autograd.grad(sum((x.float()*w.float()).sum() for x,w in zip(actual,weights)),xs)
    natural=torch.autograd.grad(sum((x.float()*w.float()).sum() for x,w in zip(expected,weights)),refs)
    # abs is nonsmooth: minute online-softmax rounding can switch sign(mu).
    # Independently differentiate dense A at the branch chosen by the actual
    # returned mean. This checks the complete VJP, including all Q/K and raw
    # abs(V) paths, without masking away any tokens or changing the kernel.
    q,k,v=[x.transpose(1,2) for x in refs]
    logits=(q.float()@k.float().transpose(-1,-2))/d**.5
    a=logits.masked_fill(~torch.ones(t,t,device=q.device,dtype=torch.bool).tril(),-torch.inf).softmax(-1)
    mean=(a@v.float()).to(v.dtype).transpose(1,2)
    raw=(a@v.float().abs()).transpose(1,2)
    chosen_sign=actual[0].detach().float().sign()
    branch_loss=(mean.float()*(weights[0].float()-weights[1].float()*chosen_sign)).sum()+(raw*weights[1].float()).sum()
    matched=torch.autograd.grad(branch_loss,refs)
    flips=actual[0].sign()!=expected[0].sign()
    result=dict(mode=MODE,t=t,d=d,forward=[shared.relative(a,b) for a,b in zip(actual,expected)],
        gradients=[shared.relative(a,b) for a,b in zip(grads,natural)],
        matched_branch_gradients=[shared.relative(a,b) for a,b in zip(grads,matched)],
        gradient_check='Dense full VJP at the fused mean sign; unrestricted dense VJP also reported, not hidden',
        max_abs=[float((a.float()-b.float()).abs().max()) for a,b in zip(actual,expected)],
        statistic_min=float(actual[1].min()),negative_fraction=float((actual[1]<0).float().mean()),
        mean_sign_disagreement=float(flips.float().mean()),mean_sign_disagreement_count=int(flips.sum()),
        flipped_dense_mean_max=float(expected[0][flips].float().abs().max()) if flips.any() else 0.,
        flipped_fused_mean_max=float(actual[0][flips].float().abs().max()) if flips.any() else 0.)
    shared.record(f'{MODE}-{t}-{d}',result)
    assert all(torch.isfinite(x).all() for x in (*actual,*grads,*matched))
    assert max(result['forward'])<.025,result
    assert max(result['matched_branch_gradients'])<.035,result
    if max(result['gradients'])>=.035:assert flips.any(),result


def test_cancellation_zero_subgradient_and_same_sign_values():
    q=k=torch.zeros(1,4,1,32,device='cuda',dtype=torch.bfloat16)
    v=torch.tensor([-1.,1.,0.,0.],device='cuda',dtype=torch.bfloat16).view(1,4,1,1).expand_as(q).clone().requires_grad_()
    mu,gap=fused_attention(q,k,v,MODE)
    assert mu[:,1:].count_nonzero()==0 and gap[:,0].count_nonzero()==0
    torch.testing.assert_close(gap[:,1],torch.ones_like(gap[:,1]),atol=0,rtol=0)
    # Only row 2 contributes: subgradient of |mu=0| is zero, derivative of
    # E|V| is sign(V)/2. Positions at V=0 have exactly zero gradient.
    dv=torch.autograd.grad(gap[:,1].float().sum(),v)[0]
    expected=torch.zeros_like(v);expected[:,0]=-.5;expected[:,1]=.5
    torch.testing.assert_close(dv,expected,atol=0,rtol=0)
    # Row 3 additionally attends an exactly-zero value, testing sign(V=0).
    _,gap=fused_attention(q,k,v,MODE)
    dv=torch.autograd.grad(gap[:,2].float().sum(),v)[0]
    assert dv[:,2:].count_nonzero()==0
    torch.testing.assert_close(dv[:,0],torch.full_like(dv[:,0],-1/3),atol=0,rtol=0)
    torch.testing.assert_close(dv[:,1],torch.full_like(dv[:,1],1/3),atol=0,rtol=0)
    for sign in (-1.,1.):
        values=torch.tensor([1.,3.],device='cuda',dtype=torch.bfloat16).view(1,2,1,1).expand(1,2,1,32).clone()*sign
        values.requires_grad_();_,gap=fused_attention(q[:,:2],k[:,:2],values,MODE)
        assert gap.count_nonzero()==0
        assert torch.autograd.grad(gap.float().sum(),values)[0].count_nonzero()==0


def test_causal_prefix_and_no_future_gradients():
    shared.test_signed_zero_and_causality(MODE)


def test_full_model_compile_and_optimizer_step():
    shared.test_model_compile_initialization_and_update(MODE)


def test_matched_initialization_zero_projection_and_muon():
    base=resume.build('none');rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    model=resume.build(MODE)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    x=torch.randint(0,256,(2,128),device='cuda')
    torch.testing.assert_close(model(x),base(x),atol=0,rtol=0)
    model(x,(x+1)%256).backward();opt,_=optimizer_for(model)
    for block in model.blocks:
        weight=block.attn.c_stat.weight
        assert weight.count_nonzero()==0 and weight.grad.norm()>0
        assert any(weight is p for g in opt.param_groups if g['kind']=='muon' for p in g['params'])
    opt.step()
    assert all(b.attn.c_stat.weight.norm()>0 for b in model.blocks)


def test_exact_optimizer_rng_restore(tmp_path):
    resume.test_exact_resume_and_fixed_gradient_update(MODE,tmp_path)


def test_config_checkpoint_and_compute_accounting():
    config=ConnectionConfig(attention_statistic=MODE,attention_kernel='fla_triton')
    assert ConnectionConfig.from_checkpoint(asdict(config))==config
    assert model_cost(config)==model_cost(ConnectionConfig(attention_statistic='jensen_silu',attention_kernel='fla_triton'))
