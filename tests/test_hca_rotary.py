"""Fused rotary norm against the unchanged nanochat operators."""
import pytest
import torch
from nanochat.gpt import norm,apply_rotary_emb
from nanochat.hca_rotary import rotary_norm
from test_hca_triton import close


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('shape',[(2,137,2,32),(1,1,4,128),(2,2048,4,128)])
def test_rotary_norm_vjp(dtype,shape):
    torch.manual_seed(644)
    x=torch.randn(shape,device='cuda',dtype=dtype,requires_grad=True)
    phase=torch.randn(1,shape[1],1,shape[-1]//2,device='cuda')
    cos,sin=phase.cos().to(dtype),phase.sin().to(dtype)
    y=rotary_norm(x,cos,sin)
    ref=norm(apply_rotary_emb(x,cos,sin))*1.2
    upstream=torch.randn_like(y)
    ga,=torch.autograd.grad(y,x,upstream,retain_graph=True)
    ge,=torch.autograd.grad(ref,x,upstream)
    close(y,ref,dtype);close(ga,ge,dtype)
    again,=torch.autograd.grad(y,x,upstream)
    torch.testing.assert_close(ga,again,rtol=0,atol=0)
