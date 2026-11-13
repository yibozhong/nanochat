"""Independent numerical and repeatability checks for the score projection."""
import pytest
import torch
from nanochat.hca_projection import score_linear


@pytest.mark.parametrize('shape', [(2, 53, 32), (1, 1, 128), (2, 2048, 512)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_projection_vjp(shape, dtype):
    torch.manual_seed(741)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(shape, device='cuda', dtype=dtype, requires_grad=True)
    w = torch.randn(shape[-1], shape[-1], device='cuda', requires_grad=True)
    y = score_linear(x, w)
    ref = torch.nn.functional.linear(x.float(), w)
    upstream = torch.randn_like(ref)
    actual = torch.autograd.grad(y, (x, w), upstream, retain_graph=True)
    expected = torch.autograd.grad(ref, (x, w), upstream)
    for a, e in zip((y, *actual), (ref, *expected)):
        assert torch.isfinite(a).all()
        rel = (a.float()-e.float()).norm()/e.float().norm()
        peak = (a.float()-e.float()).abs().max()/e.float().abs().max()
        tol = .001 if a.dtype == torch.bfloat16 else 3e-5
        assert rel < tol and peak < tol, (float(rel), float(peak))
    again = torch.autograd.grad(y, (x, w), upstream)
    for a, e in zip(actual, again):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
