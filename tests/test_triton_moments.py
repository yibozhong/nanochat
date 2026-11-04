"""PyTorch independent formula/VJP and integrated BF16 causal training checks."""
import json
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from nanochat.triton_moment_attention import fused_attention
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for
from nanochat.connection_experiment import configure_attention

@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.recompile_limit=64;configure_attention('sdpa_cudnn')

def reference(q,k,v,mode):
    # Match BF16 operands/phi and BF16 output mean, but use independent dense
    # FP32 attention math to assess errors from online softmax/Tensor Core P.
    q,k,v=[x.transpose(1,2) for x in (q,k,v)]
    t=q.shape[-2];a=(q.float()@k.float().transpose(-1,-2))/q.shape[-1]**.5
    a=a.masked_fill(~torch.ones(t,t,device=q.device,dtype=torch.bool).tril(),-torch.inf).softmax(-1)
    mu=(a@v.float()).to(v.dtype)
    if mode=='none':return mu.transpose(1,2),torch.empty(0,device=q.device)
    phi=lambda x:(F.silu(x.float()) if mode=='jensen_silu' else x.float().square()).to(v.dtype).float()
    raw=a@phi(v)
    stat=raw if mode=='second_moment' else raw-phi(mu)
    if mode=='variance':stat=stat.clamp_min(0)
    return mu.transpose(1,2),stat.to(v.dtype).transpose(1,2)

def record(name,result):
    p=Path('results/triton-numerics');p.mkdir(parents=True,exist_ok=True)
    (p/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')

def relative(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-9))

@pytest.mark.parametrize('mode',['none','second_moment','variance','jensen_silu'])
@pytest.mark.parametrize('t,d',[(17,32),(129,64),(2048,128)])
def test_operator_forward_and_gradients(mode,t,d):
    xs=[torch.randn(1,t,2,d,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    ref=[x.detach().clone().requires_grad_() for x in xs]
    out=fused_attention(*xs,mode);expected=reference(*ref,mode)
    weights=[torch.randn_like(x) for x in out]
    loss=sum((x.float()*w.float()).sum() for x,w in zip(out,weights))
    rloss=sum((x.float()*w.float()).sum() for x,w in zip(expected,weights))
    grads=torch.autograd.grad(loss,xs);rgrads=torch.autograd.grad(rloss,ref)
    result=dict(mode=mode,t=t,d=d,forward=[relative(a,b) for a,b in zip(out,expected)],
        gradients=[relative(a,b) for a,b in zip(grads,rgrads)],
        max_abs=[float((a.float()-b.float()).abs().max()) if a.numel() else 0 for a,b in zip(out,expected)])
    record(f'{mode}-{t}-{d}',result)
    assert all(torch.isfinite(x).all() for x in (*out,*grads))
    assert max(result['forward'])<.025,result
    assert max(result['gradients'])<.035,result
    # Different upstream branches must both reach Q/K/V; tests use random Wo/Wj adjoints.
    if mode!='none':
        assert all(g.norm()>0 for g in grads)

@pytest.mark.parametrize('mode',['variance','jensen_silu'])
def test_signed_zero_and_causality(mode):
    q=k=torch.zeros(1,128,1,32,device='cuda',dtype=torch.bfloat16)
    v=torch.full_like(q,3)
    _,s=fused_attention(q,k,v,mode)
    assert s[:,0].count_nonzero()==0
    assert s.float().abs().max()<.02
    if mode=='jensen_silu':
        v[:,::2]=5
        _,s=fused_attention(q,k,v,mode)
        assert s[:,1].mean()<0
    xs=[torch.randn(1,256,1,32,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    out=fused_attention(*xs,mode);pert=[x.detach().clone() for x in xs]
    for x in pert:x[:,128:]+=3
    changed=fused_attention(*pert,mode)
    for a,b in zip(out,changed):assert torch.equal(a[:,:128],b[:,:128])
    grads=torch.autograd.grad(sum(x[:,:128].float().square().sum() for x in out),xs)
    assert all(g[:,128:].count_nonzero()==0 for g in grads)

@pytest.mark.parametrize('mode',['none','variance','jensen_silu'])
def test_model_compile_initialization_and_update(mode):
    c=ConnectionConfig(depth=2,width=128,heads=1,vocab_size=256,sequence_length=128,
                       attention_kernel='fla_triton',attention_statistic=mode)
    torch.manual_seed(42);model=ConnectionModel(c).prepare()
    torch.manual_seed(42);compiled=ConnectionModel(c).prepare(compile_blocks=True)
    assert all(torch.equal(v,compiled.state_dict()[k]) for k,v in model.state_dict().items())
    with torch.no_grad():
        for b in model.blocks:
            b.attn.c_proj.weight.normal_(std=.02)
            if b.attn.c_stat is not None:b.attn.c_stat.weight.normal_(std=.02)
    compiled.load_state_dict(model.state_dict())
    x=torch.randint(0,256,(2,128),device='cuda');y=(x+1)%256
    losses=[];grads=[]
    for m in (model,compiled):
        loss=m(x,y);loss.backward();losses.append(float(loss));grads.append(torch.cat([p.grad.float().flatten() for p in m.parameters()]))
    result=dict(losses=losses,gradient_relative=relative(*grads));record('model-'+mode,result)
    assert abs(losses[0]-losses[1])<.002 and result['gradient_relative']<.02,result
    opt,_=optimizer_for(compiled);opt.step()
    assert all(torch.isfinite(p).all() for p in compiled.parameters())
