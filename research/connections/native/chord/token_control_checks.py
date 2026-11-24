"""GPU oracle for the token read/write, including the zero-speed RMSNorm derivative."""
import torch

from nanochat.chord_token_controls import GATE_EPS, reference_read, reference_write
from nanochat.chord_token_read import TokenRead
from nanochat.chord_triton import TokenWrite


def check_vjp(expected, actual, inputs, *, bounded_read=False):
    expected, actual = tuple(x for x in expected if x is not None), tuple(x for x in actual if x is not None)
    probes = [torch.randn_like(x) for x in actual]
    for x, y in zip(actual, expected):
        if bounded_read and x.dtype == torch.bfloat16:
            assert (x.float()-y.float()).norm() < .003 * y.float().norm()
        elif x.dtype == torch.bfloat16:
            torch.testing.assert_close(x.float(), y.float(), atol=.032, rtol=.012)
        else:
            torch.testing.assert_close(x, y, atol=3e-5 if bounded_read else 3e-6, rtol=2e-5)
    lhs = torch.autograd.grad(actual, inputs, probes)
    rhs = torch.autograd.grad(expected, inputs, probes)
    for x, y in zip(lhs, rhs):
        diff = (x.float()-y.float()).norm() / y.float().norm().clamp_min(1e-7)
        assert torch.isfinite(x).all() and diff < .015, (x.shape, diff.item())
    return lhs, rhs


def check_token_controls(*, bounded=False):
    torch.manual_seed(714)
    D = 512
    def rand(*shape):
        return torch.randn(shape, device='cuda')
    for terminal in (False, True):
        h, v = rand(2, 71, D).bfloat16(), rand(2, 71, D).bfloat16()
        v[:, 0] = 0
        v[:, 1:8] *= .0001
        v[:, 8:16] *= .01
        v[:, 16:24] *= 16
        w = rand(3 if terminal else 6, 2, D) * (.1/D**.5)
        mu = rand(1) * .2
        steps = torch.tensor([-2.8, -2.3], device='cuda')
        beta, gains = (None, None) if terminal else (rand(1), rand(2))
        args = (h, v, w, mu, steps, beta, gains)
        leaves = [x.requires_grad_() for x in args if x is not None]
        q, p = TokenRead.apply(*args, bounded)
        gx, gy = check_vjp(reference_read(*args, bounded), (q, p), leaves, bounded_read=bounded)
        for group in (slice(0, 1), slice(1, 8), slice(8, 16), slice(16, 24), slice(24, None)):
            g, ref = gx[1][:, group].float(), gy[1][:, group].float()
            assert (g-ref).norm() < .02 * ref.norm().clamp_min(1e-6)
        with torch.no_grad():
            if bounded:
                inv = ((h.float().square() + v.float().square()).mean(-1) / 2 + GATE_EPS).rsqrt()
                z = (h.float() @ w[0, 0] + v.float() @ w[0, 1]) * inv
            else:
                gate_v = torch.nn.functional.rms_norm(v.float(), (D,), eps=GATE_EPS)
                z = h.float() @ w[0, 0] + gate_v @ w[0, 1]
            raw_mu = z + mu
            expected_mu = 3 * (raw_mu / 3).tanh() if bounded else raw_mu
            torch.testing.assert_close(p[..., 0], expected_mu, atol=1e-6, rtol=2e-5)
            q2, p2 = TokenRead.apply(8*h if bounded else h, 8*v, *args[2:], bounded)
            torch.testing.assert_close(p[:, 24:], p2[:, 24:], atol=2e-6, rtol=2e-5)
            if bounded:
                q2, _ = TokenRead.apply(h, 8*v, *args[2:], True)
            assert (q[:, 24:].float()-q2[:, 24:].float()).abs().max() > .05
        f = rand(*h.shape).bfloat16().requires_grad_()
        controls = p.detach().requires_grad_()
        args = (h, v, f, controls)
        check_vjp(reference_write(*args, 1e-6, terminal),
                  TokenWrite.apply(*args, 1e-6, terminal), args)
        if bounded:
            # Drive gates into both tails while leaving unsaturated tokens for gradient checks.
            with torch.no_grad():
                w.mul_(80)
            args = (h, v, w, mu, steps, beta, gains)
            q, p = TokenRead.apply(*args, True)
            check_vjp(reference_read(*args, True), (q, p), leaves, bounded_read=True)
            assert (p[..., 0].abs() <= 3).all() and (p[..., 1:] >= 0).all()
            assert (p[..., 1:4] <= 1).all() and (p[..., 4:] <= 2).all()
            if terminal:
                assert (p[..., 3:] == .5).all()
