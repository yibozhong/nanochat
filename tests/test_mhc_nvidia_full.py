"""Independent math, layout, fused-gradient and full-model NVIDIA gates."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from nanochat.nvidia_connections import NvidiaHyperConnection, NvidiaHyperHead, interleave_phi
from nanochat.nvidia_projection import nvidia_ops
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for
from nanochat.engine import KVCache


def relative(a, b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()


def record(name, data):
    p=Path('results/nvidia-full-checks');p.mkdir(parents=True,exist_ok=True)
    (p/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')


def reference_mix(x, phi, alpha, beta):
    flat=x.flatten(-2)
    h=F.linear(flat/(flat.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt(),phi)
    pre=(h[...,:4]*alpha[0]+beta[:4]).sigmoid()
    post=2*(h[...,4:8]*alpha[1]+beta[4:8]).sigmoid()
    logp=(h[...,8:]*alpha[2]+beta[8:]).reshape(*x.shape[:2],4,4)
    for _ in range(20):
        logp=logp-torch.logsumexp(logp,dim=-1,keepdim=True)
        logp=logp-torch.logsumexp(logp,dim=-2,keepdim=True)
    return pre,post,logp.exp()


@pytest.mark.parametrize('width',[128,512])
@pytest.mark.parametrize('fused',[False,True])
def test_nvidia_full_pipeline_fp64(width,fused):
    torch.manual_seed(352)
    m=NvidiaHyperConnection(width,4,0,use_tf32=True,fused_grad_acc=fused).cuda();m.reset_parameters()
    x=torch.randn(2,128,width,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    branch,res,mix=m.read(x)
    # A dependent branch checks the required fused-buffer backward ordering.
    y=torch.tanh(branch.float()).to(x.dtype)
    out=m.write(y,res,mix)
    p=torch.randn_like(out).float()/out.numel()
    q=torch.randn_like(branch).float()/branch.numel()
    actual=torch.autograd.grad((out.float()*p).sum()+(branch.float()*q).sum(),(x,*m.parameters()))
    xr=x.detach().double().requires_grad_()
    params=[v.detach().double().requires_grad_() for v in m.parameters()]
    phi,bias,scales=params
    pre,post,comb=reference_mix(xr,phi,scales,bias)
    br=(xr*pre.unsqueeze(-2)).sum(-1)
    yr=br.tanh()
    expected=torch.einsum('btcn,btnm->btcm',xr,comb)+yr.unsqueeze(-1)*post.unsqueeze(-2)
    reference=torch.autograd.grad((expected*p.double()).sum()+(br*q.double()).sum(),(xr,*params))
    errors=dict(output=relative(out,expected),branch=relative(branch,br),
                gradients=[relative(a,b) for a,b in zip(actual,reference)])
    record(f'pipeline-d{width}-fused{int(fused)}',errors)
    assert errors['output']<.01 and errors['branch']<.01,errors
    assert max(errors['gradients'])<.03,errors
    assert all(torch.isfinite(a).all() and a.norm()>0 for a in actual)


def test_nvidia_scale_sinkhorn_fp64_and_extreme_logits():
    torch.manual_seed(93);ops=nvidia_ops()
    h=torch.randn(128,32,device='cuda',requires_grad=True)
    a=torch.tensor([.13,.31,.7],device='cuda',requires_grad=True)
    b=torch.randn(1,24,device='cuda',requires_grad=True)
    ms=(torch.rand(128,device='cuda')+.1).requires_grad_()
    pre,post,res=ops.mhc_fused_scale(h,a,b,ms,4)
    comb=ops.mhc_fused_sinkhorn(res.reshape(1,128,4,4),4,iters=20)
    hr,ar,br,mr=[v.detach().double().requires_grad_() for v in (h,a,b,ms)]
    s=hr[:,:24]/(mr[:,None]+torch.finfo(torch.float32).eps).sqrt()
    pr=(s[:,:4]*ar[0]+br[:,:4]).sigmoid();po=2*(s[:,4:8]*ar[1]+br[:,4:8]).sigmoid()
    lr=(s[:,8:]*ar[2]+br[:,8:]).reshape(1,128,4,4)
    for _ in range(20):
        lr=lr-torch.logsumexp(lr,-1,keepdim=True)
        lr=lr-torch.logsumexp(lr,-2,keepdim=True)
    cr=lr.exp();probes=[torch.randn_like(v) for v in (pre,post,comb)]
    ga=torch.autograd.grad(sum((v*p).sum() for v,p in zip((pre,post,comb),probes)),(h,a,b,ms))
    gr=torch.autograd.grad(sum((v*p.double()).sum() for v,p in zip((pr,po,cr),probes)),(hr,ar,br,mr))
    errors=dict(output=[relative(x,y) for x,y in zip((pre,post,comb),(pr,po,cr))],
                gradients=[relative(x,y) for x,y in zip(ga,gr)],
                row_residual_20=(comb.sum(-1)-1).abs().max().item(),
                fp64_row_residual_20=(cr.sum(-1)-1).abs().max().item())
    record('scale-sinkhorn',errors)
    assert max(errors['output'])<2e-5 and max(errors['gradients'])<1e-4,errors
    # A finite 20-iteration algorithm is not an exact doubly-stochastic solve.
    # Check its row residual against the same-iteration FP64 reference instead
    # of assuming arbitrary logits converge to a preset residual in 20 steps.
    torch.testing.assert_close(comb.sum(-1).double(),cr.sum(-1),atol=2e-6,rtol=2e-5)
    assert (comb.sum(-2)-1).abs().max()<1e-5
    converged=ops.mhc_fused_sinkhorn(res.detach().reshape(1,128,4,4),4,iters=200)
    assert (converged.sum(-1)-1).abs().max()<1e-5
    z=(torch.randn(2,64,4,4,device='cuda')*100).requires_grad_()
    out=ops.mhc_fused_sinkhorn(z,4,iters=20)
    grad=torch.autograd.grad((out*torch.randn_like(out)).sum(),z)[0]
    assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    assert (out.sum(-2)-1).abs().max()<1e-4


def test_nvidia_final_head_fp64():
    torch.manual_seed(84)
    m=NvidiaHyperHead(512,4,use_tf32=False).cuda();m.reset_parameters()
    x=torch.randn(1,128,512,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    out=m(x);p=torch.randn_like(out).float()/out.numel()
    ga=torch.autograd.grad((out.float()*p).sum(),(x,*m.parameters()))
    xr=x.detach().double().requires_grad_();params=[v.detach().double().requires_grad_() for v in m.parameters()]
    phi,scale,bias=params;flat=xr.flatten(-2)
    raw=F.linear(flat/(flat.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt(),phi)
    pre=(raw*scale+bias).sigmoid();expected=(xr*pre.unsqueeze(-2)).sum(-1)
    gr=torch.autograd.grad((expected*p.double()).sum(),(xr,*params))
    errors=dict(output=relative(out,expected),gradients=[relative(a,b) for a,b in zip(ga,gr)])
    record('head',errors)
    assert errors['output']<.005 and max(errors['gradients'])<.02,errors


@pytest.mark.parametrize('fused',[False,True])
def test_nvidia_model_no_deepseek_causality_cache_cce_and_layout(fused):
    torch.manual_seed(42)
    opts=dict(variant='mhc',width=128,heads=2,depth=2,vocab_size=128,sequence_length=2048)
    old=ConnectionModel(ConnectionConfig(**opts)).prepare()
    torch.manual_seed(42)
    model=ConnectionModel(ConnectionConfig(**opts,connection_backend='nvidia',fused_connection_grad=fused)).prepare()
    # Only projection columns differ in storage. Their effective initialization
    # and all other parameters match the existing model exactly.
    for k,v in model.state_dict().items():
        ref=old.state_dict()[k]
        if k.endswith('.phi'):ref=interleave_phi(ref)
        torch.testing.assert_close(v,ref,atol=0,rtol=0)
    optimizer_for(model)
    with torch.no_grad():
        for block in model.blocks:
            block.attn.c_proj.weight.normal_(std=.03);block.mlp.c_proj.weight.normal_(std=.03)
    ids=torch.randint(0,128,(1,2048),device='cuda');changed=ids.clone();changed[:,1024:]=(changed[:,1024:]+17)%128
    with patch('nanochat.manifold_connections.official_ops',side_effect=AssertionError('DeepSeek mHC called')):
        with torch.no_grad():
            a=model(ids);b=model(changed)
            torch.testing.assert_close(a[:,:1024],b[:,:1024],atol=0,rtol=0)
            cache=KVCache(1,2,32,64,2,'cuda',torch.bfloat16)
            pre=model(ids[:,:7],kv_cache=cache);one=model(ids[:,7:8],kv_cache=cache);full=model(ids[:,:8])
            torch.testing.assert_close(pre,full[:,:7],atol=.004,rtol=.03)
            torch.testing.assert_close(one,full[:,7:],atol=.004,rtol=.03)
        x,y=ids[:,:128].contiguous(),ids[:,1:129].contiguous()
        loss=model(x,y);dense=model(x,y,dense=True)
        torch.testing.assert_close(loss,dense,atol=2e-4,rtol=2e-4)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        before={k:p.grad.clone() for k,p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        model.config.activation_checkpoint=True
        again=model(x,y);again.backward()
        torch.testing.assert_close(loss,again,atol=2e-5,rtol=2e-5)
        for k,p in model.named_parameters():
            assert relative(p.grad,before[k])<.02,k


def test_fused_buffer_matches_unfused_and_repeated_backward():
    torch.manual_seed(13)
    m=NvidiaHyperConnection(128,4,0,use_tf32=False,fused_grad_acc=True).cuda();m.reset_parameters()
    n=copy.deepcopy(m);n.fused_grad_acc=False
    x=torch.randn(1,64,128,4,device='cuda',dtype=torch.bfloat16)
    p=torch.randn_like(x)
    def run(layer):
        xi=x.detach().clone().requires_grad_();br,res,mix=layer.read(xi)
        out=layer.write(torch.sin(br.float()).to(x.dtype),res,mix)
        loss=(out.float()*p).mean()
        g=torch.autograd.grad(loss,(xi,*layer.parameters()),retain_graph=True)
        repeat=torch.autograd.grad(loss,(xi,*layer.parameters()))
        assert max(relative(a,b) for a,b in zip(g,repeat))<1e-4
        return out,g
    a,ga=run(m);b,gb=run(n)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    errors=[relative(a,b) for a,b in zip(ga,gb)];record('fused-vs-unfused',errors)
    assert max(errors)<.01,errors


def test_tma_allocator_in_fresh_context():
    import contextvars
    from triton.runtime import _allocation
    # Initialize TE's process-global flag in the current context first.
    m=NvidiaHyperConnection(128,4,0,use_tf32=False).cuda();m.reset_parameters()
    x=torch.randn(1,32,128,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    br,res,mix=m.read(x);m.write(br,res,mix).float().mean().backward()
    def fresh_context():
        assert isinstance(_allocation._allocator.get(),_allocation.NullAllocator)
        xi=x.detach().clone().requires_grad_()
        branch,residual,mixes=m.read(xi)
        assert not isinstance(_allocation._allocator.get(),_allocation.NullAllocator)
        out=m.write(branch,residual,mixes)
        grads=torch.autograd.grad(out.float().square().mean(),(xi,*m.parameters()))
        assert all(torch.isfinite(g).all() for g in grads)
    contextvars.Context().run(fresh_context)
