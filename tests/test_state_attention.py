"""Independent FP64 oracle and locality/activation checks for the Triton core."""
import pytest
import torch

from nanochat.state_attention import state_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU kernel gate')


def reference(q, k, v):
    logits = torch.einsum('nihd,njhd->nhij', q, k) / q.shape[-1]**.5
    return torch.einsum('nhij,njhd->nihd', logits.softmax(-1), v)


@pytest.mark.parametrize('n,nq,nk,h,d,amp', [
    (1,1,4,1,64,1.), (7,1,4,3,96,1.), (13,4,5,2,128,1.),
    (5,4,5,1,64,0.), (3,4,5,2,128,1e-4)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_all_input_gradients_against_fp64(n,nq,nk,h,d,amp,dtype):
    torch.manual_seed(43)
    args = [(torch.randn(shape,device='cuda',dtype=dtype)*amp).requires_grad_()
            for shape in ((n,nq,h,d),(n,nk,h,d),(n,nk,h,d))]
    out = state_attention(*args); probe = torch.randn_like(out)
    grad = torch.autograd.grad(out,args,probe)
    oracle = [x.detach().double().requires_grad_() for x in args]
    expected = reference(*oracle)
    expected_grad = torch.autograd.grad(expected,oracle,probe.double())
    errors = []
    for a,b in zip((out,*grad),(expected,*expected_grad)):
        assert torch.isfinite(a).all()
        relative = ((a.double()-b).norm()/b.norm().clamp_min(1e-12)).item()
        errors.append(relative)
        assert relative < (.008 if dtype==torch.bfloat16 else 2e-5), errors
    again = torch.autograd.grad(state_attention(*args),args,probe)
    assert all(torch.equal(x,y) for x,y in zip(grad,again))


def test_token_head_locality_and_saved_inputs_only():
    torch.manual_seed(99)
    q = torch.randn(9,4,3,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    k = torch.randn(9,5,3,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    v = torch.randn_like(k,requires_grad=True)
    saved=[]
    def pack(t):saved.append(t);return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):out=state_attention(q,k,v)
    assert len(saved)==3 and all(a is b for a,b in zip(saved,(q,k,v)))
    kv = k.detach().clone(); kv[4,:,1] += 3
    changed = state_attention(q,kv,v)
    mask=torch.ones(out.shape[:-1],device='cuda',dtype=torch.bool);mask[4,:,1]=False
    assert torch.equal(out[mask],changed[mask])
    gs=torch.autograd.grad(out[4,:,1].float().sum(),(q,k,v))
    for g in gs:
        untouched=torch.ones(g.shape[:-1],device='cuda',dtype=torch.bool);untouched[4,:,1]=False
        assert torch.count_nonzero(g[untouched])==0


def test_noncontiguous_inputs_and_output_gradient():
    torch.manual_seed(123)
    q=torch.randn(7,2,4,64,device='cuda').transpose(1,2).requires_grad_()
    k=torch.randn(7,2,5,64,device='cuda').transpose(1,2).requires_grad_()
    v=torch.randn_like(k).requires_grad_()
    probe=torch.randn(7,2,4,64,device='cuda').transpose(1,2)
    a=state_attention(q,k,v);b=reference(q,k,v)
    ga=torch.autograd.grad(a,(q,k,v),probe);gb=torch.autograd.grad(b,(q,k,v),probe)
    for x,y in zip((a,*ga),(b,*gb)):torch.testing.assert_close(x,y,rtol=2e-4,atol=3e-6)
