"""FP32 HCA score projection using Tensor Core input decomposition.

Master weights and outputs stay FP32. No global matmul precision setting is
changed. Split-K uses a deterministic reduction, with no floating atomics.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable

PRECISION = 'bf16x3'
GROUP_M = 8

@tr.jit
def _mm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        AM: tl.constexpr, AK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
        SPLIT: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr, BT: tl.constexpr,
        PRECISION: tl.constexpr, GROUP: tl.constexpr):
    # Visit nearby M/N tiles together so the long activation matrix stays in
    # L2 while its output columns are processed. Output tiles remain disjoint.
    pid=tl.program_id(0)
    nm,nn=tl.cdiv(M,BM),tl.cdiv(N,BC)
    group=pid//(GROUP*nn)
    first=group*GROUP
    size=tl.minimum(nm-first,GROUP)
    pm=first+(pid%(GROUP*nn))%size
    pn=(pid%(GROUP*nn))//size
    m = pm*BM + tl.arange(0, BM)
    n = pn*BC + tl.arange(0, BC)
    part = tl.program_id(1)
    span: tl.constexpr = tl.cdiv(K, SPLIT*BT)*BT
    kk = tl.arange(0, BT)
    acc = tl.zeros((BM, BC), tl.float32)
    for base in range(part*span, tl.minimum(K, (part+1)*span), BT):
        k = base + kk
        a = tl.load(A+m[:, None]*AM+k[None, :]*AK,
                    (m[:, None] < M) & (k[None, :] < K), 0).to(tl.float32)
        b = tl.load(B+k[:, None]*BK+n[None, :]*BN,
                    (k[:, None] < K) & (n[None, :] < N), 0).to(tl.float32)
        if PRECISION == 'tf32x3':
            acc += tl.dot(a, b, input_precision='tf32x3')
        else:
            # Keep high/residual products; omit only residual*residual.
            # BF16 inputs need no residual term. Accuracy has an FP64 oracle.
            ah, bh = a.to(tl.bfloat16), b.to(tl.bfloat16)
            if A.dtype.element_ty != tl.bfloat16:
                al = (a-ah.to(tl.float32)).to(tl.bfloat16)
                acc += tl.dot(al, bh)
            if B.dtype.element_ty != tl.bfloat16:
                bl = (b-bh.to(tl.float32)).to(tl.bfloat16)
                acc += tl.dot(ah, bl)
            acc += tl.dot(ah, bh)
    tl.store(C+part*M*N+m[:, None]*N+n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


def matmul(a, b, split=1):
    m, k = a.shape
    kb, n = b.shape
    assert k == kb and a.is_cuda and b.is_cuda
    out = torch.empty((split, m, n), device=a.device, dtype=torch.float32)
    _mm[(tr.cdiv(m, 64)*tr.cdiv(n, 64), split)](
        a, b, out, m, n, k, *a.stride(), *b.stride(), split, 64, 64, 32, PRECISION, GROUP_M,
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
