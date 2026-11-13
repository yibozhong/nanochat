"""Independent checks for the confidence boundary estimator and compact VJPs."""
from dataclasses import replace
import pytest
import torch
from nanochat.hca import balanced_compression_loss, confidence_output, weighted_attention
from nanochat.hca_triton import attention
from test_hca import config, model
from test_hca_triton import close


def test_confidence_estimator_against_explicit_surrogate():
    torch.manual_seed(19)
    y = torch.randn(2, 37, 2, 16, device='cuda', dtype=torch.float64, requires_grad=True)
    logits = torch.randn(2, 37, device='cuda', dtype=torch.float64, requires_grad=True)
    p = logits.sigmoid()
    hard = (p >= .5).double()
    out = confidence_output(y, hard, p)
    selected = torch.where(hard.bool(), p, 1-p)
    ref = y*(1+selected-selected.detach())[:, :, None, None]
    torch.testing.assert_close(out, y, rtol=0, atol=0)
    upstream = torch.randn_like(y)
    ga = torch.autograd.grad(out, (y, logits), upstream, retain_graph=True)
    ge = torch.autograd.grad(ref, (y, logits), upstream)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, rtol=1e-12, atol=1e-12)


def test_balanced_rate_gradient_corrects_both_directions():
    for hard_rate, sign in [(0., -1), (1/16, 0), (.5, 1)]:
        soft_count = torch.tensor(10., device='cuda', requires_grad=True)
        loss = balanced_compression_loss(soft_count, hard_rate*100, 100, 16)
        g, = torch.autograd.grad(loss, soft_count)
        assert g.sign().item() == sign


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('kind', ['single', 'every', 'mixed'])
def test_compact_attention_vjps(dtype, kind):
    torch.manual_seed(787)
    b, t, h, d, w = 2, 137, 2, 32, 17
    xs = [torch.randn(b,t,h,d,device='cuda',dtype=dtype,requires_grad=True) for _ in range(5)]
    starts = (torch.rand(b,t,device='cuda') < .1).float()
    if kind == 'single': starts.zero_()
    if kind == 'every': starts.fill_(1)
    starts[:,0] = 1
    y = attention(*xs, starts, w, 0)
    pos = torch.arange(t, device='cuda')
    local = (pos[:,None] >= pos) & (pos[:,None]-pos < w)
    closed = torch.cat((starts[:,1:], starts.new_zeros(b,1)),1)
    memory = (pos[:,None] > pos).float()[None]*closed[:,None] + torch.eye(t,device='cuda')[None]
    weights = torch.cat((local[None].expand(b,-1,-1),memory),-1)
    eligible = torch.cat((local,pos[:,None] >= pos),-1)[None].expand(b,-1,-1)
    q,k,v,pk,pv = xs
    ref = weighted_attention(q,torch.cat((k,pk),1),torch.cat((v,pv),1),weights,eligible)
    close(y,ref,dtype)
    upstream = torch.randn_like(y)
    ga = torch.autograd.grad(y,xs,upstream,retain_graph=True)
    ge = torch.autograd.grad(ref,xs,upstream)
    for a,e in zip(ga,ge): close(a,e,dtype)
    again = torch.autograd.grad(y,xs,upstream)
    for a,e in zip(ga,again): torch.testing.assert_close(a,e,rtol=0,atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_confidence_model_gradients_causality_and_cache(dtype):
    c = config(hca_dynamic_gradient='confidence', hca_noise=False, hca_block_size=16)
    ref = model(c,dtype).eval()
    fast = model(replace(c,hca_backend='triton'),dtype).eval()
    fast.load_state_dict(ref.state_dict())
    ids = torch.randint(0,64,(2,53),device='cuda')
    yr,yf = ref(ids),fast(ids)
    close(yf,yr,dtype)
    # CE alone, without the rate penalty, must train the router.
    ref(ids,(ids+1)%64,dense=True).backward()
    fast(ids,(ids+1)%64,dense=True).backward()
    for (name,p),(_,r) in zip(fast.named_parameters(),ref.named_parameters()):
        assert (p.grad is None)==(r.grad is None),name
        if p.grad is not None: close(p.grad,r.grad,dtype,pooling=True)
    for block in fast.blocks:
        assert block.attn.boundary.weight.grad.abs().max()>0
    with torch.no_grad():
        altered=ids.clone();altered[:,19:]=(altered[:,19:]+1)%64
        torch.testing.assert_close(fast(altered)[:,:19],yf[:,:19],rtol=0,atol=0)
        cache=fast.make_kv_cache(2)
        parts=[];lo=0
        for n in (3,1,11,2,16,7,13):
            parts.append(fast(ids[:,lo:lo+n],kv_cache=cache));lo+=n
        close(torch.cat(parts,1),yf,dtype,pooling=True)
        saved=cache.state_dict();restored=fast.make_kv_cache(2);restored.load_state_dict(saved)
        assert restored.statistics()==cache.statistics()
    x=torch.randn(1,53,32,device='cuda',dtype=dtype,requires_grad=True)
    out=fast.blocks[0].attn(x,None,(fast.cos[:,:53],fast.sin[:,:53]),(-1,0),None)
    gx,=torch.autograd.grad(out[:,:19].float().square().sum(),x)
    assert gx[:,19:].count_nonzero()==0
