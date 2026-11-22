"""NeMo-style per-token depth softmax over separate Chord endpoint tensors."""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(xs, nw, pw, out, stats, n, D: tl.constexpr, K: tl.constexpr, R: tl.constexpr, EPS: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, D)
    offsets = rows[:, None] * D + cols[None, :]
    valid = rows[:, None] < n
    weight = tl.load(nw + cols) * tl.load(pw + cols)
    scores = ()
    norms = ()
    maximum = tl.full((R,), -float('inf'), tl.float32)
    for j in tl.static_range(K):
        x = tl.load(xs[j] + offsets, mask=valid, other=0.).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
        score = tl.sum(x * weight[None, :], 1) * inv
        norms += (inv,)
        scores += (score,)
        maximum = tl.maximum(maximum, score)
    denom = tl.full((R,), 0., tl.float32)
    for j in tl.static_range(K):
        denom += tl.exp(scores[j] - maximum)
    out_value = tl.full((R, D), 0., tl.float32)
    for j in tl.static_range(K):
        p = tl.exp(scores[j] - maximum) / denom
        x = tl.load(xs[j] + offsets, mask=valid, other=0.).to(tl.float32)
        out_value += p[:, None] * x
        tl.store(stats + j*n + rows, p, mask=rows<n)
        tl.store(stats + (K+j)*n + rows, norms[j], mask=rows<n)
        tl.store(stats + (2*K+j)*n + rows, scores[j], mask=rows<n)
    tl.store(out + offsets, out_value, mask=valid)


@triton.jit
def _backward(xs, dxs, nw, pw, dy, stats, partial, n, D: tl.constexpr, K: tl.constexpr, R: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, D)
    offsets = rows[:, None] * D + cols[None, :]
    valid = rows[:, None] < n
    weight = tl.load(nw + cols) * tl.load(pw + cols)
    grad = tl.load(dy + offsets, mask=valid, other=0.).to(tl.float32)
    products = ()
    mean = tl.full((R,), 0., tl.float32)
    for j in tl.static_range(K):
        x = tl.load(xs[j] + offsets, mask=valid, other=0.).to(tl.float32)
        dot = tl.sum(x*grad, 1)
        products += (dot,)
        p = tl.load(stats + j*n + rows, mask=rows<n, other=0.)
        mean += p * dot
    dw = tl.full((R, D), 0., tl.float32)
    for j in tl.static_range(K):
        x = tl.load(xs[j] + offsets, mask=valid, other=0.).to(tl.float32)
        p = tl.load(stats + j*n + rows, mask=rows<n, other=0.)
        inv = tl.load(stats + (K+j)*n + rows, mask=rows<n, other=0.)
        s = tl.load(stats + (2*K+j)*n + rows, mask=rows<n, other=0.)
        ds = p * (products[j] - mean)
        key_grad = ds * inv
        radial = key_grad * (s * inv / D)
        dx = p[:, None]*grad + key_grad[:, None]*weight[None, :] - radial[:, None]*x
        tl.store(dxs[j] + offsets, dx, mask=valid)
        dw += key_grad[:, None]*x
    tl.store(partial + tl.program_id(0)*D + cols, tl.sum(dw, 0))


class _EndpointMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nw, pw, eps, *xs):
        # All endpoints and q have contiguous [B,T,D] storage in production.
        xs = tuple(x.contiguous() for x in xs)
        n, d = xs[0].numel() // xs[0].size(-1), xs[0].size(-1)
        out = torch.empty_like(xs[0])
        stats = torch.empty((3, len(xs), n), device=out.device, dtype=torch.float32)
        _forward[(triton.cdiv(n, 4),)](xs, nw, pw, out, stats, n, d, len(xs), 4, eps)
        ctx.save_for_backward(nw, pw, stats, *xs)
        return out

    @staticmethod
    def backward(ctx, grad):
        nw, pw, stats, *xs = ctx.saved_tensors
        k, n, d = len(xs), stats.size(-1), grad.size(-1)
        rows = 8
        parts = torch.empty((triton.cdiv(n, rows), d), device=grad.device, dtype=torch.float32)
        dx = tuple(torch.empty_like(x) for x in xs)
        _backward[(parts.size(0),)](tuple(xs), dx, nw, pw, grad.contiguous(), stats, parts, n, d, k, rows)
        dw = parts.sum(0)
        return dw * pw, dw * nw, None, *dx


def mix_endpoints(q, history, reader):
    # Preserve each endpoint's autograd edge without rebuilding or saving a
    # packed copy of all earlier endpoints whenever a new block finishes.
    return _EndpointMix.apply(reader.norm_weight, reader.proj_weight, reader.eps, *history, q)
