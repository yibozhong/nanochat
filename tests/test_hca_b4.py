"""The additionally authorized B=4 configuration, including unfinished chunks."""
from dataclasses import replace
import pytest
import torch
from test_hca import config, model
from test_hca_triton import close


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_fixed_b4_vjp_causality_split_cache(dtype):
    c=config('fixed_hca', hca_block_size=4, hca_backend='reference')
    ref=model(c,dtype).eval(); fast=model(replace(c,hca_backend='triton'),dtype).eval()
    fast.load_state_dict(ref.state_dict())
    ids=torch.randint(0,64,(2,67),device='cuda');targets=(ids+1)%64
    expected=ref(ids);actual=fast(ids);close(actual,expected,dtype)
    ref(ids,targets,dense=True).backward();fast(ids,targets,dense=True).backward()
    for (name,p),(_,q) in zip(fast.named_parameters(),ref.named_parameters()):
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:close(p.grad,q.grad,dtype,pooling=True)
    with torch.no_grad():
        changed=ids.clone();changed[:,21:]=(changed[:,21:]+1)%64
        torch.testing.assert_close(fast(changed)[:,:21],actual[:,:21],rtol=0,atol=0)
        cache=fast.make_kv_cache(2);parts=[];start=0
        for n in (1,2,4,7,16,3,34):
            parts.append(fast(ids[:,start:start+n],kv_cache=cache));start+=n
        close(torch.cat(parts,1),actual,dtype,pooling=True)


def test_fixed_b4_2k_split_cache():
    m=model(config('fixed_hca',hca_block_size=4,hca_backend='triton',sequence_length=2048),torch.bfloat16).eval()
    ids=torch.randint(0,64,(1,2048),device='cuda')
    with torch.no_grad():
        expected=m(ids);cache=m.make_kv_cache(1)
        parts=[];start=0
        for n in (3,513,1,1024,507):
            parts.append(m(ids[:,start:start+n],kv_cache=cache));start+=n
        close(torch.cat(parts,1),expected,torch.bfloat16,pooling=True)
