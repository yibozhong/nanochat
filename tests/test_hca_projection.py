"""Independent numerical and repeatability checks for the score projection."""
import pytest
import torch
from nanochat.hca_projection import score_linear, matmul


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
        if a.dtype == torch.bfloat16:
            # A single BF16 rounding threshold can change normalized peak by
            # ~0.2% even when FP32 relative error is <1e-6. Validate the actual
            # pre-cast dX against an independent FP64 oracle, then establish
            # exact BF16 rounding for both implementations. This retains the
            # strict FP32 tolerance instead of loosening a BF16 peak threshold.
            with torch.no_grad():
                dy = upstream.reshape(-1, w.shape[0])
                pre = matmul(dy, w).view_as(x)
                ref_pre = (dy @ w).view_as(x)
                oracle = (dy.double() @ w.double()).view_as(x).float()
                torch.testing.assert_close(a, pre.to(dtype), rtol=0, atol=0)
                torch.testing.assert_close(e, ref_pre.to(dtype), rtol=0, atol=0)
                for value in (pre, ref_pre):
                    rel = (value-oracle).norm()/oracle.norm()
                    peak = (value-oracle).abs().max()/oracle.abs().max()
                    assert rel < 3e-5 and peak < 3e-5, (float(rel),float(peak))
            continue
        rel = (a.float()-e.float()).norm()/e.float().norm()
        peak = (a.float()-e.float()).abs().max()/e.float().abs().max()
        tol = 3e-5
        assert rel < tol and peak < tol, (float(rel), float(peak))
    again = torch.autograd.grad(y, (x, w), upstream)
    for a, e in zip(actual, again):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
