"""Three-branch BF16 attention: independent centered oracle and full-model checks."""
from dataclasses import asdict
import pytest
import torch
from nanochat.triton_moment_attention import fused_attention
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for
from nanochat.connection_cost import model_cost
import test_triton_moments as shared
import test_triton_resume as resume

setup=shared.setup
MODE='variance_third'


def formula(q,k,v,rounded=True):
    q,k,v=[x.transpose(1,2) for x in (q,k,v)]
    dtype=torch.float32 if rounded else q.dtype
    t=q.shape[-2];logits=(q.to(dtype)@k.to(dtype).transpose(-1,-2))/q.shape[-1]**.5
    a=logits.masked_fill(~torch.ones(t,t,device=q.device,dtype=torch.bool).tril(),-torch.inf).softmax(-1)
    cast=lambda x:x.to(v.dtype).to(dtype) if rounded else x
    mu=cast(a@v.to(dtype))
    r2=a@cast(v.to(dtype).square());r3=a@cast(v.to(dtype).pow(3))
    var=(r2-cast(mu.square())).clamp_min(0)
    c3=(r3-cast(mu.pow(3)))-3*mu*(r2-cast(mu.square()))
    if rounded:
        c3=torch.where(torch.arange(t,device=q.device)[None,None,:,None]>0,c3,0.)
    return tuple((x.to(v.dtype) if rounded else x).transpose(1,2) for x in (mu,var,c3))


def centered_oracle(q,k,v):
    means=[];variances=[];thirds=[]
    for i in range(q.shape[1]):
        values=v[:,:i+1]
        a=(torch.einsum('bhd,bshd->bhs',q[:,i],k[:,:i+1])/q.shape[-1]**.5).softmax(-1)
        a=a.transpose(1,2).unsqueeze(-1);mu=(a*values).sum(1)
        centered=values-mu[:,None]
        means.append(mu);variances.append((a*centered.square()).sum(1));thirds.append((a*centered.pow(3)).sum(1))
    return tuple(torch.stack(xs,1) for xs in (means,variances,thirds))


@pytest.mark.parametrize('compiled',[False,True])
def test_raw_formula_against_centered_oracle(compiled):
    dtype=torch.float32 if compiled else torch.float64
    xs=[torch.randn(2,16,2,8,device='cuda',dtype=dtype,requires_grad=True) for _ in range(3)]
    fn=torch.compile(formula,fullgraph=True,dynamic=False) if compiled else formula
    actual=fn(*xs,rounded=False);refs=[x.detach().double().requires_grad_() for x in xs]
    expected=centered_oracle(*refs);weights=[torch.randn_like(x) for x in actual]
    for a,e in zip(actual,expected):torch.testing.assert_close(a.double(),e,atol=2e-5 if compiled else 3e-12,rtol=3e-6 if compiled else 3e-12)
    ga=torch.autograd.grad(sum((x*w).sum() for x,w in zip(actual,weights)),xs)
    ge=torch.autograd.grad(sum((x*w.double()).sum() for x,w in zip(expected,weights)),refs)
    for a,e in zip(ga,ge):assert (a.double()-e).norm()/e.norm().clamp_min(1e-12)<(3e-6 if compiled else 3e-12)


def check_operator(t,d,branch=None):
    xs=[torch.randn(1,t,2,d,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    refs=[x.detach().clone().requires_grad_() for x in xs]
    actual=fused_attention(*xs,MODE);expected=formula(*refs)
    weights=[torch.randn_like(x) if branch is None or i==branch else torch.zeros_like(x) for i,x in enumerate(actual)]
    ga=torch.autograd.grad(sum((x.float()*w.float()).sum() for x,w in zip(actual,weights)),xs)
    ge=torch.autograd.grad(sum((x.float()*w.float()).sum() for x,w in zip(expected,weights)),refs)
    old=fused_attention(*(x.detach() for x in xs),'variance')
    row=dict(t=t,d=d,branch=branch,forward=[shared.relative(a,e) for a,e in zip(actual,expected)],
        gradients=[shared.relative(a,e) for a,e in zip(ga,ge)],
        variance_path=[shared.relative(a,e) for a,e in zip(actual[:2],old)],
        first_token_third_max=float(actual[2][:,0].abs().max()),
        third_rms=float(actual[2].float().square().mean().sqrt()),third_max=float(actual[2].abs().max()))
    shared.record(f'third-{t}-{d}-{branch}',row)
    assert all(torch.isfinite(x).all() for x in (*actual,*ga))
    assert max(row['forward'])<.025 and max(row['gradients'])<.035,row
    assert max(row['variance_path'])<.01,row


@pytest.mark.parametrize('t,d',[(17,32),(129,64),(2048,128)])
def test_fused_all_outputs_and_qkv_gradients(t,d):check_operator(t,d)


@pytest.mark.parametrize('branch',[0,1,2])
def test_independent_branch_vjp(branch):check_operator(129,64,branch)


def test_zero_signed_translation_and_cubic_scaling():
    q=k=torch.zeros(1,3,1,32,device='cuda',dtype=torch.bfloat16)
    v=torch.tensor([0.,0.,3.],device='cuda',dtype=torch.bfloat16).view(1,3,1,1).expand_as(q).contiguous()
    mu,var,c3=fused_attention(q,k,v,MODE)
    assert c3[:,0].count_nonzero()==0
    torch.testing.assert_close(c3[:,-1],torch.full_like(c3[:,-1],2),atol=0,rtol=0)
    shifted=fused_attention(q,k,v+1,MODE)
    torch.testing.assert_close(shifted[1],var,atol=0,rtol=0)
    torch.testing.assert_close(shifted[2],c3,atol=0,rtol=0)
    for scale in (-1.,2.):
        other=fused_attention(q,k,v*scale,MODE)
        torch.testing.assert_close(other[1],var*scale**2,atol=0,rtol=0)
        torch.testing.assert_close(other[2],c3*scale**3,atol=0,rtol=0)
    constant=fused_attention(q,k,torch.ones_like(v)*3,MODE)
    assert constant[1].count_nonzero()==0 and constant[2].count_nonzero()==0


def test_singleton_arbitrary_bf16_values_have_zero_third_and_vjp():
    xs=[torch.randn(2,129,2,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    with torch.no_grad():xs[2].mul_(4)
    c3=fused_attention(*xs,MODE)[2]
    assert c3[:,0].count_nonzero()==0
    grads=torch.autograd.grad((c3[:,0].float()*torch.randn_like(c3[:,0]).float()).sum(),xs)
    assert all(g.count_nonzero()==0 for g in grads)


def test_real_2k_causal_prefix_and_no_future_vjp():
    xs=[torch.randn(1,2048,1,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    actual=fused_attention(*xs,MODE);pert=[x.detach().clone() for x in xs]
    for x in pert:x[:,1024:]+=3
    other=fused_attention(*pert,MODE)
    for a,e in zip(actual,other):assert torch.equal(a[:,:1024],e[:,:1024])
    grads=torch.autograd.grad(sum(x[:,:1024].float().square().mean() for x in actual),xs)
    assert all(g[:,1024:].count_nonzero()==0 for g in grads)


def test_shared_initial_weights_zero_projections_and_muon():
    base=resume.build('variance');rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    model=resume.build(MODE)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in base.parameters())==2*128**2
    x=torch.randint(0,256,(2,128),device='cuda')
    torch.testing.assert_close(model(x),base(x),atol=0,rtol=0)
    model(x,(x+1)%256).backward();opt,_=optimizer_for(model)
    for b in model.blocks:
        for weight in (b.attn.c_stat.weight,b.attn.c_third.weight):
            assert weight.count_nonzero()==0 and weight.grad.norm()>0
            assert any(weight is p for g in opt.param_groups if g['kind']=='muon' for p in g['params'])
    opt.step()
    assert all(b.attn.c_third.weight.norm()>0 for b in model.blocks)


def test_full_model_compile_and_recompute_gradients():
    config=dict(depth=2,width=128,heads=1,vocab_size=256,sequence_length=128,
                attention_kernel='fla_triton',attention_statistic=MODE)
    model=ConnectionModel(ConnectionConfig(**config)).prepare()
    with torch.no_grad():
        for b in model.blocks:
            for weight in (b.attn.c_proj.weight,b.attn.c_stat.weight,b.attn.c_third.weight):weight.normal_(std=.02)
    x=torch.randint(0,256,(2,128),device='cuda');y=(x+1)%256
    loss=model(x,y);loss.backward();grad=torch.cat([p.grad.float().flatten() for p in model.parameters()])
    rows=[]
    for recompute in (False,True):
        other=ConnectionModel(ConnectionConfig(**config,activation_checkpoint=recompute)).prepare(compile_blocks=True)
        other.load_state_dict(model.state_dict());other_loss=other(x,y);other_loss.backward()
        other_grad=torch.cat([p.grad.float().flatten() for p in other.parameters()])
        row=dict(recompute=recompute,ce_difference=abs(float(other_loss-loss)),gradient_relative=shared.relative(other_grad,grad));rows.append(row)
        assert row['ce_difference']<.002 and row['gradient_relative']<.02,row
        opt,_=optimizer_for(other);opt.step();assert all(torch.isfinite(p).all() for p in other.parameters())
    shared.record('model-third',dict(comparisons=rows))


def test_exact_optimizer_rng_restore(tmp_path):resume.test_exact_resume_and_fixed_gradient_update(MODE,tmp_path)


def test_config_and_cost():
    c=ConnectionConfig(attention_statistic=MODE,attention_kernel='fla_triton')
    assert ConnectionConfig.from_checkpoint(asdict(c))==c
    a=model_cost(c)['components_forward']
    b=model_cost(ConnectionConfig(attention_statistic='variance',attention_kernel='fla_triton'))['components_forward']
    for k in ('statistics_dense_matmuls','statistics_projection'):assert a[k]==2*b[k]
    with pytest.raises(ValueError):ConnectionConfig(attention_statistic=MODE)
