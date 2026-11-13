"""FP32 HCA score projection using TF32x3 Tensor Core products.

Master weights and outputs stay FP32. No global matmul precision setting is
changed. Split-K uses a deterministic reduction, with no floating atomics.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable


@tr.jit
def _mm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        AM: tl.constexpr, AK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
        SPLIT: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr, BT: tl.constexpr):
    m = tl.program_id(0)*BM + tl.arange(0, BM)
    n = tl.program_id(1)*BC + tl.arange(0, BC)
    part = tl.program_id(2)
    span: tl.constexpr = tl.cdiv(K, SPLIT*BT)*BT
    kk = tl.arange(0, BT)
    acc = tl.zeros((BM, BC), tl.float32)
    for base in range(part*span, tl.minimum(K, (part+1)*span), BT):
        k = base + kk
        a = tl.load(A+m[:, None]*AM+k[None, :]*AK,
                    (m[:, None] < M) & (k[None, :] < K), 0).to(tl.float32)
        b = tl.load(B+k[:, None]*BK+n[None, :]*BN,
                    (k[:, None] < K) & (n[None, :] < N), 0).to(tl.float32)
        acc += tl.dot(a, b, input_precision='tf32x3')
    tl.store(C+part*M*N+m[:, None]*N+n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


def matmul(a, b, split=1):
    m, k = a.shape
    kb, n = b.shape
    assert k == kb and a.is_cuda and b.is_cuda
    out = torch.empty((split, m, n), device=a.device, dtype=torch.float32)
    _mm[(tr.cdiv(m, 64), tr.cdiv(n, 64), split)](
        a, b, out, m, n, k, *a.stride(), *b.stride(), split, 64, 64, 32,
        num_warps=4)
    return out[0] if split == 1 else out.sum(0)


class _ScoreLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        x = x.contiguous()
        ctx.save_for_backward(x, weight)
        y = matmul(x.reshape(-1, x.shape[-1]), weight.T)
        return y.view(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        dy = grad.reshape(-1, weight.shape[0]).contiguous()
        dx = matmul(dy, weight).view_as(x).to(x.dtype)
        dw = matmul(dy.T, x.reshape(-1, x.shape[-1]),
                    split=min(16, tr.cdiv(dy.shape[0], 1024))).to(weight.dtype)
        return dx, dw


def score_linear(x, weight):
    return _ScoreLinear.apply(x, weight)
