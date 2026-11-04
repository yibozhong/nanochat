"""Effective-support attention: independent dense oracle and integration checks."""
from dataclasses import asdict
import pytest
import torch
from nanochat.triton_moment_attention import fused_attention
from nanochat.moment_attention import SupportGate
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for
from nanochat.connection_cost import model_cost
import test_triton_moments as shared
import test_triton_resume as resume

setup=shared.setup
MODE='variance_support'


def log_support_reference(q,k):
    q,k=q.transpose(1,2).float(),k.transpose(1,2).float()
    t=q.shape[-2]
    scores=(q@k.transpose(-1,-2))/q.shape[-1]**.5
    a=scores.masked_fill(~torch.ones(t,t,device=q.device,dtype=torch.bool).tril(),-torch.inf).softmax(-1)
    return -a.square().sum(-1).log().transpose(1,2)


def reference(q,k,v):
    return (*shared.reference(q,k,v,'variance'),log_support_reference(q,k))


def test_support_softmax_derivative_against_autograd_fp64():
    scores=torch.randn(2,4,16,device='cuda',dtype=torch.float64,requires_grad=True)
    a=scores.softmax(-1);logn=-a.square().sum(-1).log();w=torch.randn_like(logn)
    actual=torch.autograd.grad((logn*w).sum(),scores)[0]
    expected=2*w[...,None]*a*(1-logn.exp()[...,None]*a)
    torch.testing.assert_close(actual,expected,atol=2e-15,rtol=2e-12)


def check_operator(t,d,branch=None):
    xs=[torch.randn(1,t,2,d,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    refs=[x.detach().clone().requires_grad_() for x in xs]
    actual=fused_attention(*xs,MODE);expected=reference(*refs)
    weights=[torch.randn_like(x) if branch is None or i==branch else torch.zeros_like(x) for i,x in enumerate(actual)]
    ga=torch.autograd.grad(sum((x.float()*w.float()).sum() for x,w in zip(actual,weights)),xs)
    ge=torch.autograd.grad(sum((x.float()*w.float()).sum() for x,w in zip(expected,weights)),refs)
    old=fused_attention(*(x.detach() for x in xs),'variance')
    row=dict(t=t,d=d,branch=branch,forward=[shared.relative(a,e) for a,e in zip(actual,expected)],
        gradients=[shared.relative(a,e) for a,e in zip(ga,ge)],
        log_support_max_error=float((actual[2]-expected[2]).abs().max()),
        variance_path=[shared.relative(a,e) for a,e in zip(actual[:2],old)])
    shared.record(f'support-{t}-{d}-{branch}',row)
    assert all(torch.isfinite(x).all() for x in (*actual,*ga))
    assert max(row['forward'][:2])<.025 and max(row['gradients'])<.035,row
    assert row['log_support_max_error']<1e-4,row
    assert all(torch.equal(a,e) for a,e in zip(actual[:2],old)),row
    if branch==2:
        assert ga[2].count_nonzero()==0 and ga[0].norm()>0 and ga[1].norm()>0


@pytest.mark.parametrize('t,d',[(17,32),(129,64),(2048,128)])
def test_all_outputs_and_qkv_gradients(t,d):check_operator(t,d)


@pytest.mark.parametrize('branch',[0,1,2])
def test_independent_branch_vjp(branch):check_operator(129,64,branch)


def test_uniform_concentrated_and_singleton():
    q=torch.zeros(2,257,4,32,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(q)
    logn=fused_attention(q,q,v,MODE)[2]
    expected=torch.arange(1,258,device='cuda').float().log()[None,:,None].expand_as(logn)
    torch.testing.assert_close(logn,expected,atol=3e-6,rtol=0)
    q.fill_(1);k=torch.zeros_like(q);k[:,0]=20
    logn=fused_attention(q,k,v,MODE)[2]
    assert logn.abs().max()<1e-6
    xs=[torch.randn_like(q,requires_grad=True) for _ in range(3)]
    logn=fused_attention(*xs,MODE)[2]
    assert logn[:,0].count_nonzero()==0
    grads=torch.autograd.grad(logn[:,0].sum(),xs)
    assert all(g.count_nonzero()==0 for g in grads)


def test_online_rescaling_head_independence_and_value_independence():
    q=torch.randn(2,513,4,32,device='cuda',dtype=torch.bfloat16)
    k=torch.randn_like(q)
    # Different heads and successive tiles force large changes of online max.
    q[:,:,:,0]=1
    k[:,:,:,0]=torch.arange(513,device='cuda')[:,None]/4
    k[:,:,:,1:]*=torch.tensor([.25,1.,2.,4.],device='cuda')[None,None,:,None]
    v=torch.randn_like(q)
    logn=fused_attention(q,k,v,MODE)[2];expected=log_support_reference(q,k)
    err=float((logn-expected).abs().max());shared.record('support-rescale',dict(max_abs=err))
    assert err<1e-4
    assert torch.equal(logn,fused_attention(q,k,v*4+3,MODE)[2])
    changed=k.clone();changed[:,:,2]*=2
    other=fused_attention(q,changed,v,MODE)[2]
    assert torch.equal(logn[:,:,[0,1,3]],other[:,:,[0,1,3]])
    neff=logn.exp();visible=torch.arange(1,514,device='cuda')[None,:,None]
    assert (neff>=1-1e-5).all() and (neff<=visible*(1+1e-5)).all()


def test_real_2k_causality_and_no_future_vjp():
    xs=[torch.randn(1,2048,2,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    actual=fused_attention(*xs,MODE);pert=[x.detach().clone() for x in xs]
    for x in pert:x[:,1024:]+=3
    other=fused_attention(*pert,MODE)
    for a,e in zip(actual,other):assert torch.equal(a[:,:1024],e[:,:1024])
    grads=torch.autograd.grad(sum(x[:,:1024].float().square().mean() for x in actual),xs)
    assert all(g[:,1024:].count_nonzero()==0 for g in grads)


def test_shared_initial_weights_identity_and_optimizer_groups():
    base=resume.build('variance');rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    model=resume.build(MODE)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in base.parameters())==4
    x=torch.randint(0,256,(2,128),device='cuda');y=(x+1)%256
    assert torch.equal(model(x),base(x))
    opt,_=optimizer_for(model)
    for b in model.blocks:
        assert b.attn.c_stat.weight.count_nonzero()==0
        assert any(b.attn.c_stat.weight is p for g in opt.param_groups if g['kind']=='muon' for p in g['params'])
        gate=b.attn.support_gate
        assert torch.equal(gate.coefficients(torch.randn(2,128,1,device='cuda')),torch.ones(2,128,1,device='cuda'))
        for p in gate.parameters():
            assert p.count_nonzero()==0 and p.dtype==torch.float32
            assert any(p is z for g in opt.param_groups if g['kind']=='adamw' and g['weight_decay']==0 for z in g['params'])
    for _ in range(3):
        opt.zero_grad(set_to_none=True);model(x,y).backward();opt.step()
    for b in model.blocks:
        assert b.attn.c_stat.weight.norm()>0
        assert all(p.grad.norm()>0 for p in b.attn.support_gate.parameters())


def test_gate_before_dense_projection_and_compiled_parameter_vjp():
    gate=SupportGate(4,'cuda')
    with torch.no_grad():gate.a.copy_(torch.tensor([-.5,.1,.5,1.],device='cuda'));gate.b.fill_(.1)
    var=torch.randn(2,17,4,32,device='cuda',requires_grad=True)
    logn=torch.rand(2,17,4,device='cuda',requires_grad=True)
    weight=torch.randn(128,128,device='cuda')
    expected=torch.cat([var[:,:,h]*2*torch.sigmoid(gate.a[h]*logn[:,:,h]+gate.b[h])[...,None] for h in range(4)],-1)@weight.T
    actual=torch.compile(gate,fullgraph=True)(var,logn).flatten(-2)@weight.T
    torch.testing.assert_close(actual,expected,atol=3e-5,rtol=1e-5)
    args=(var,logn,gate.a,gate.b);w=torch.randn_like(actual)
    ga=torch.autograd.grad((actual*w).sum(),args);ge=torch.autograd.grad((expected*w).sum(),args)
    for a,e in zip(ga,ge):assert shared.relative(a,e)<2e-5


def test_gated_attention_complete_gradient_against_dense():
    xs=[torch.randn(1,129,4,32,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    refs=[x.detach().clone().requires_grad_() for x in xs]
    a=torch.tensor([-.3,.2,.5,1.],device='cuda',requires_grad=True);b=torch.zeros_like(a,requires_grad=True)
    ar=a.detach().clone().requires_grad_();br=b.detach().clone().requires_grad_()
    actual=fused_attention(*xs,MODE);expected=reference(*refs)
    def combine(m,aa,bb):return m[0].float()+m[1].float()*2*torch.sigmoid(aa*m[2]+bb)[...,None]
    actual=combine(actual,a,b);expected=combine(expected,ar,br);w=torch.randn_like(actual)
    ga=torch.autograd.grad((actual*w).sum(),(*xs,a,b));ge=torch.autograd.grad((expected*w).sum(),(*refs,ar,br))
    errors=[shared.relative(x,y) for x,y in zip(ga,ge)]
    shared.record('support-gated-vjp',dict(gradient_relative=errors))
    assert max(errors)<.035 and all(g.norm()>0 for g in ga)


def test_full_model_compile_and_recompute_gradients():
    config=dict(depth=2,width=128,heads=4,vocab_size=256,sequence_length=128,
                attention_kernel='fla_triton',attention_statistic=MODE)
    model=ConnectionModel(ConnectionConfig(**config)).prepare()
    with torch.no_grad():
        for b in model.blocks:
            for weight in (b.attn.c_proj.weight,b.attn.c_stat.weight):weight.normal_(std=.02)
            b.attn.support_gate.a.copy_(torch.tensor([-.3,.2,.5,1.],device='cuda'))
            b.attn.support_gate.b.fill_(.1)
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
    shared.record('model-support',dict(comparisons=rows))


def test_exact_optimizer_rng_restore(tmp_path):resume.test_exact_resume_and_fixed_gradient_update(MODE,tmp_path)


def test_config_and_cost():
    c=ConnectionConfig(attention_statistic=MODE,attention_kernel='fla_triton')
    assert ConnectionConfig.from_checkpoint(asdict(c))==c
    other=ConnectionConfig(attention_statistic='variance',attention_kernel='fla_triton')
    assert model_cost(c)['components_forward']==model_cost(other)['components_forward']
    with pytest.raises(ValueError):ConnectionConfig(attention_statistic=MODE)
