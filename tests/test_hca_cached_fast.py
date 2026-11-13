"""Training/inference arithmetic at local tiles and compressed-chunk boundaries."""
import pytest
import torch
from nanochat.hca_triton import attention, cached_attention, cached_pool, prefix_pool
from nanochat.hca import LayerCache
from test_hca_triton import close


@pytest.mark.parametrize('ratio', [0, 4, 16])
def test_cached_attention_same_inputs(ratio):
    torch.manual_seed(982)
    t,h,d,w = 537,4,128,256
    q,k,v,pk,pv = [torch.randn(1,t,h,d,device='cuda',dtype=torch.bfloat16) for _ in range(5)]
    starts = (torch.rand(1,t,device='cuda') < .1).float()
    if ratio: starts = (torch.arange(t,device='cuda') % ratio == 0).float()[None]
    starts[:,0] = 1
    with torch.no_grad():
        full = attention(q,k,v,pk,pv,starts,w,ratio,compact=not ratio)
        for pos in (0,1,3,4,15,16,31,32,63,64,65,127,128,255,256,257,511,512,536):
            end = starts[0,1:pos+1].nonzero().flatten()
            sk,sv = pk[0,end],pv[0,end]
            local = slice(max(0,pos-w+1),pos+1)
            actual = cached_attention(q[:,pos:pos+1],k[0,local],v[0,local],
                sk,sv,pk[0,pos:pos+1],pv[0,pos:pos+1],pos,w,ratio)
            close(actual,full[:,pos:pos+1],torch.bfloat16)
            torch.testing.assert_close(actual,full[:,pos:pos+1],rtol=0,atol=0)


@pytest.mark.parametrize('ratio', [0, 4, 16])
def test_cached_pool_scan_rounding(ratio):
    torch.manual_seed(193)
    t,h,d = 137,4,128
    k,v = [torch.randn(1,t,h,d,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
    score=torch.randn(1,t,h,d,device='cuda')*2
    starts=(torch.rand(1,t,device='cuda') < .07).float()
    if ratio:starts=(torch.arange(t,device='cuda')%ratio == 0).float()[None]
    starts[:,0]=1
    with torch.no_grad():
        pk,pv=prefix_pool(k,v,score,starts,ratio)
        state=LayerCache();ak=[];av=[]
        for i in range(t):
            x,y=cached_pool(k[:,i],v[:,i],score[:,i],starts[:,i],state,ratio)
            ak.append(x);av.append(y);state.position+=1
        torch.testing.assert_close(torch.stack(ak,1),pk,rtol=0,atol=0)
        torch.testing.assert_close(torch.stack(av,1),pv,rtol=0,atol=0)
