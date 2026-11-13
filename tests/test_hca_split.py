"""Independent VJP coverage of query-split completed-summary backward."""
import pytest
import torch
from nanochat import hca_triton as hca
from nanochat.hca import weighted_attention
from test_hca_triton import close


@pytest.mark.parametrize('d',[32,128])
@pytest.mark.parametrize('ratio',[4,16])
@pytest.mark.parametrize('splits',[2,4,8])
def test_fixed_summary_splits(d,ratio,splits,monkeypatch):
    monkeypatch.setattr(hca,'FIXED_DKV_SPLITS',splits)
    monkeypatch.setattr(hca,'FIXED_SMALL_SPLITS',splits)
    monkeypatch.setattr(hca,'FIXED_BWD_TILE',(64,64,4))
    monkeypatch.setattr(hca,'FIXED_BWD_STAGES',1)
    torch.manual_seed(807)
    b,t,h,w=2,137,2,17
    xs=[torch.randn(b,t,h,d,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(5)]
    pos=torch.arange(t,device='cuda')
    starts=(pos%ratio==0).float()[None].expand(b,-1)
    y=hca.attention(*xs,starts,w,ratio)
    local=(pos[:,None]>=pos)&(pos[:,None]-pos<w)
    closed=torch.cat((starts[:,1:],starts.new_zeros(b,1)),1)
    memory=(pos[:,None]>pos).float()[None]*closed[:,None]+torch.eye(t,device='cuda')[None]
    weights=torch.cat((local[None].expand(b,-1,-1),memory),-1)
    eligible=torch.cat((local,pos[:,None]>=pos),-1)[None].expand(b,-1,-1)
    q,k,v,pk,pv=xs
    ref=weighted_attention(q,torch.cat((k,pk),1),torch.cat((v,pv),1),weights,eligible)
    upstream=torch.randn_like(y)
    ga=torch.autograd.grad(y,xs,upstream,retain_graph=True)
    ge=torch.autograd.grad(ref,xs,upstream)
    close(y,ref,torch.bfloat16)
    for a,e in zip(ga,ge):close(a,e,torch.bfloat16)
    again=torch.autograd.grad(y,xs,upstream)
    for a,e in zip(ga,again):torch.testing.assert_close(a,e,rtol=0,atol=0)
