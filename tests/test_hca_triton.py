"""GPU-only mathematical and integration checks, run on the Condor worker."""
from dataclasses import replace

import pytest
import torch

from nanochat.hca import pool_step, weighted_attention
from nanochat.hca_triton import prefix_pool, attention
from test_hca import config, model


def close(actual, expected, dtype, *, pooling=False):
    # Relative L2 plus a scale-normalized max error handles BF16 rounding and
    # cancellation near zero. Neither threshold is adjusted by the launcher.
    a, e = actual.detach().float(), expected.detach().float()
    assert torch.isfinite(a).all() and torch.isfinite(e).all()
    rel = (a-e).norm() / e.norm().clamp_min(1e-7)
    peak = (a-e).abs().max() / e.abs().max().clamp_min(1.)
    tol = .02 if dtype == torch.bfloat16 else (2e-4 if pooling else 3e-5)
    assert rel <= tol and peak <= tol, (float(rel), float(peak), tol)


def gates(b, t, kind, ratio=16):
    if kind == 'fixed':
        g = (torch.arange(t, device='cuda') % ratio == 0).float()[None].repeat(b, 1)
    elif kind == 'single':
        g = torch.zeros(b, t, device='cuda')
    elif kind == 'every':
        g = torch.ones(b, t, device='cuda')
    else:
        g = (torch.rand(b, t, device='cuda') < .2).float()
    g[:, 0] = 1
    return g.requires_grad_()


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('kind,ratio', [('fixed',16), ('fixed',3), ('single',0), ('every',0), ('mixed',0)])
def test_pool_all_vjps(dtype, kind, ratio):
    torch.manual_seed(932)
    b, t, h, d = 2, 137, 2, 24  # cross scan tiles, partial block and channel tile
    k = torch.randn(b,t,h,d,device='cuda',dtype=dtype,requires_grad=True)
    v = torch.randn_like(k,requires_grad=True)
    s = (torch.randn(k.shape,device='cuda')*2).requires_grad_()
    g = gates(b,t,kind,ratio or 16)
    pk, pv = prefix_pool(k,v,s,g,ratio)
    state = tuple(torch.zeros(b,h,d,device='cuda') for _ in range(3))
    rk, rv = [], []
    for i in range(t):
        state, ki, vi = pool_step(k[:,i],v[:,i],s[:,i],state,g[:,i])
        rk.append(ki); rv.append(vi)
    rk,rv = torch.stack(rk,1).to(dtype),torch.stack(rv,1).to(dtype)
    close(pk,rk,dtype,pooling=True);close(pv,rv,dtype,pooling=True)
    uk,uv = torch.randn_like(pk),torch.randn_like(pv)
    inputs = (k,v,s) if ratio else (k,v,s,g)
    actual = torch.autograd.grad((pk,pv),inputs,(uk,uv),retain_graph=True)
    ref = torch.autograd.grad((rk,rv),inputs,(uk,uv))
    for a,e in zip(actual,ref):close(a,e,dtype,pooling=True)
    # No atomics: repeated backward on identical inputs must be bitwise equal.
    again = torch.autograd.grad((pk,pv),inputs,(uk,uv))
    for a,e in zip(actual,again):torch.testing.assert_close(a,e,rtol=0,atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('kind', ['fixed','single','every','mixed'])
def test_attention_all_vjps_including_zero_gate(dtype,kind):
    torch.manual_seed(443)
    b,t,h,d,w=2,73,2,32,17
    xs=[torch.randn(b,t,h,d,device='cuda',dtype=dtype,requires_grad=True) for _ in range(5)]
    q,k,v,pk,pv=xs
    g=gates(b,t,kind)
    ratio=16 if kind=='fixed' else 0
    y=attention(*xs,g,w,ratio)
    pos=torch.arange(t,device='cuda')
    local=(pos[:,None]>=pos)&(pos[:,None]-pos<w)
    closed=torch.cat((g[:,1:],g.new_zeros(b,1)),1)
    memory=(pos[:,None]>pos).float()[None]*closed[:,None,:]+torch.eye(t,device='cuda')[None]
    weights=torch.cat((local[None].expand(b,-1,-1),memory),-1)
    eligible=torch.cat((local,pos[:,None]>=pos),-1)[None].expand(b,-1,-1)
    ref=weighted_attention(q,torch.cat((k,pk),1),torch.cat((v,pv),1),weights,eligible)
    close(y,ref,dtype)
    upstream=torch.randn_like(y)
    inputs=xs if ratio else xs+[g]
    ga=torch.autograd.grad(y,inputs,upstream,retain_graph=True)
    ge=torch.autograd.grad(ref,inputs,upstream)
    for a,e in zip(ga,ge):close(a,e,dtype)
    if kind in ('single','mixed'):
        assert ga[-1][g==0].abs().max()>1e-5, 'Hard compaction erased zero-gate STE gradients'
    again=torch.autograd.grad(y,inputs,upstream)
    for a,e in zip(ga,again):torch.testing.assert_close(a,e,rtol=0,atol=0)


@pytest.mark.parametrize('mode',['fixed_hca','dynamic_hca'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_model_ce_gradients_causality_and_cache(mode,dtype):
    ref=model(config(mode,hca_backend='reference'),dtype).eval()
    fast=model(replace(ref.config,hca_backend='triton'),dtype).eval()
    fast.load_state_dict(ref.state_dict())
    ids=torch.randint(0,64,(2,53),device='cuda')
    target=(ids+1)%64
    yr,yf=ref(ids),fast(ids)
    close(yf,yr,dtype)
    lr=ref(ids,target,dense=True);lf=fast(ids,target,dense=True)
    lr.backward();lf.backward()
    for (name,p),(_,r) in zip(fast.named_parameters(),ref.named_parameters()):
        assert (p.grad is None)==(r.grad is None),name
        if p.grad is not None:close(p.grad,r.grad,dtype,pooling=True)
    with torch.no_grad():
        altered=ids.clone();altered[:,19:]=(altered[:,19:]+1)%64
        torch.testing.assert_close(fast(altered)[:,:19],yf[:,:19],rtol=0,atol=0)
        cache=fast.make_kv_cache(2)
        pieces=[];start=0
        for n in (3,1,11,2,16,7,13):
            pieces.append(fast(ids[:,start:start+n],kv_cache=cache));start+=n
        close(torch.cat(pieces,1),yf,dtype,pooling=True)
    a=fast.blocks[0].attn
    x=torch.randn(1,53,32,device='cuda',dtype=dtype,requires_grad=True)
    y=a(x,None,(fast.cos[:,:53],fast.sin[:,:53]),(-1,0),None)
    gx,=torch.autograd.grad(y[:,:19].float().square().sum(),x)
    assert gx[:,19:].count_nonzero()==0


def test_long_open_chunk_extreme_pooling_and_noise_recompute():
    c=config(hca_backend='triton',sequence_length=320,hca_noise=True,activation_checkpoint=True)
    m=model(c).train()
    for block in m.blocks:
        with torch.no_grad():block.attn.boundary.weight.zero_();block.attn.boundary.bias.fill_(-1000.)
    ids=torch.randint(0,64,(1,257),device='cuda')
    state=torch.cuda.get_rng_state()
    loss=m(ids,(ids+1)%64,dense=True);loss.backward()
    expected={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True);torch.cuda.set_rng_state(state)
    loss2=m(ids,(ids+1)%64,dense=True);loss2.backward()
    torch.testing.assert_close(loss,loss2,rtol=0,atol=0)
    for n,p in m.named_parameters():
        if n in expected:torch.testing.assert_close(p.grad,expected[n],rtol=0,atol=0)
    m.eval()
    with torch.no_grad():
        expected=m(ids);cache=m.make_kv_cache()
        actual=torch.cat((m(ids[:,:101],kv_cache=cache),m(ids[:,101:],kv_cache=cache)),1)
        close(actual,expected,torch.float32,pooling=True)
        assert cache.statistics()['compressed_entries']==[[1],[1]]
        assert cache.statistics()['open_chunk_tokens']==[[257],[257]]
    k=torch.randn(1,257,2,32,device='cuda',requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    s=(torch.randn_like(k)*1000).requires_grad_()
    g=gates(1,257,'single')
    pk,pv=prefix_pool(k,v,s,g)
    w=(10*torch.tanh(s/10)).softmax(1)
    close(pk[:,-1],(w*k).sum(1),torch.float32,pooling=True)
    close(pv[:,-1],(w*v).sum(1),torch.float32,pooling=True)
    for grad in torch.autograd.grad(pk.sum()+pv.sum(),(k,v,s,g)):
        assert torch.isfinite(grad).all()


def test_sequences_shorter_than_or_exactly_one_chunk():
    torch.manual_seed(8)
    for t in (1,15,16,17):
        for mode in ('fixed_hca','dynamic_hca'):
            ref=model(config(mode),torch.float32).eval()
            fast=model(replace(ref.config,hca_backend='triton'),torch.float32).eval()
            fast.load_state_dict(ref.state_dict())
            ids=torch.randint(0,64,(1,t),device='cuda')
            yr,yf=ref(ids),fast(ids)
            close(yf,yr,torch.float32,pooling=True)
            yr.square().mean().backward();yf.square().mean().backward()
            for p,r in zip(fast.parameters(),ref.parameters()):
                if p.grad is not None:close(p.grad,r.grad,torch.float32,pooling=True)
