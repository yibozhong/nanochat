"""Scalar FP32 projection and confidence gradients, including production shapes."""
import pytest
import torch
from nanochat.hca_routing import router_linear,confidence_gradient


def close(a,e):
    delta=a.float()-e.float()
    assert delta.norm()/e.float().norm().clamp_min(1e-7) < 3e-5
    assert delta.abs().max()/e.float().abs().max().clamp_min(1.) < 3e-5


@pytest.mark.parametrize('shape',[(2,53,32),(1,1,512),(16,2048,512)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_router_projection_and_vjps(shape,dtype):
    torch.manual_seed(843);torch.backends.cuda.matmul.allow_tf32=False
    x=torch.randn(*shape,device='cuda',dtype=dtype,requires_grad=True)
    w=torch.randn(1,shape[-1],device='cuda',requires_grad=True)
    b=torch.randn(1,device='cuda',requires_grad=True)
    actual=router_linear(x,w,b)
    expected=torch.nn.functional.linear(x.float(),w,b).squeeze(-1)
    close(actual,expected)
    g=torch.randn_like(actual)
    ga=torch.autograd.grad(actual,(x,w,b),g,retain_graph=True)
    ge=torch.autograd.grad(expected,(x,w,b),g)
    torch.testing.assert_close(ga[0],ge[0],rtol=0,atol=0)
    for a,e in zip(ga[1:],ge[1:]):close(a,e)
    again=torch.autograd.grad(actual,(x,w,b),g)
    for a,e in zip(ga,again):torch.testing.assert_close(a,e,rtol=0,atol=0)
    if shape[1]>1:
        split=torch.cat([router_linear(x[:,:7],w,b),router_linear(x[:,7:],w,b)],1)
        torch.testing.assert_close(split,actual,rtol=0,atol=0)


@pytest.mark.parametrize('shape',[(2,53,2,16),(16,2048,4,128)])
def test_confidence_product_reduction(shape):
    torch.manual_seed(617)
    g,y=[torch.randn(*shape,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
    actual=confidence_gradient(g,y)
    expected=(g.double()*y.double()).sum((-1,-2))
    close(actual,expected)
