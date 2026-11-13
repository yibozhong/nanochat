"""Chunked packed GEMMs plus fused gate/softcap/CE and exact analytic VJP.

Uses the compute-gradients-in-forward strategy of fused linear CE (also used
by Liger-Kernel), with a reusable bounded logits buffer and FP32 dW accumulation.
No full sequence-by-vocabulary logits, probability, or gradient tensor survives.
The implementation is independent; the CUDA matrix multiplications use cuBLAS.
"""
import torch
from torch.autograd.function import once_differentiable
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _transform(z, g, A: tl.constexpr, S: tl.constexpr):
    t = libdevice.tanh(g)
    gate = 1. + A * t
    cap = libdevice.tanh(z * gate / S)
    return S * cap, t, gate, cap


@tr.jit
def _row_ce(Z, Y, Loss, N: tl.constexpr, V: tl.constexpr, TOTAL: tl.constexpr,
            A: tl.constexpr, S: tl.constexpr, GRAD: tl.constexpr, BV: tl.constexpr):
    row = tl.program_id(0)
    v = tl.arange(0, BV)
    z = tl.load(Z + row * (2*V) + v, v < V, 0).to(tl.float32)
    g = tl.load(Z + row * (2*V) + V + v, v < V, 0).to(tl.float32)
    logits, t, gate, cap = _transform(z, g, A, S)
    logits = tl.where(v < V, logits, -float('inf'))
    mx = tl.max(logits, 0)
    ex = tl.exp(logits - mx)
    den = tl.sum(ex, 0)
    y = tl.load(Y + row)
    target = tl.sum(tl.where(v == y, logits, 0.), 0)
    tl.store(Loss + row, mx + tl.log(den) - target)
    if GRAD:
        e = (ex / den - (v == y).to(tl.float32)) / TOTAL
        d = e * (1. - cap * cap)
        tl.store(Z + row * (2*V) + v, d * gate, v < V)
        tl.store(Z + row * (2*V) + V + v, d * z * A * (1. - t*t), v < V)


@tr.jit
def _partial_lse(Z, P, N: tl.constexpr, V: tl.constexpr, PARTS: tl.constexpr,
                 A: tl.constexpr, S: tl.constexpr, BV: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    v = part * BV + tl.arange(0, BV)
    z = tl.load(Z + row * (2*V) + v, v < V, 0).to(tl.float32)
    g = tl.load(Z + row * (2*V) + V + v, v < V, 0).to(tl.float32)
    logits, _, _, _ = _transform(z, g, A, S)
    logits = tl.where(v < V, logits, -float('inf'))
    mx = tl.max(logits, 0)
    lse = mx + tl.log(tl.sum(tl.exp(logits - mx), 0))
    tl.store(P + row * PARTS + part, lse)


@tr.jit
def _finish_lse(Z, Y, P, LSE, Loss, N: tl.constexpr, V: tl.constexpr,
                PARTS: tl.constexpr, A: tl.constexpr, S: tl.constexpr,
                BP: tl.constexpr, BR: tl.constexpr):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    p = tl.arange(0, BP)
    lp = tl.load(P + row[:,None]*PARTS + p[None,:], (row[:,None]<N)&(p[None,:]<PARTS), -float('inf'))
    mx = tl.max(lp, 1)
    lse = mx + tl.log(tl.sum(tl.exp(lp-mx[:,None]), 1))
    y = tl.load(Y+row, row<N, 0)
    z = tl.load(Z + row*(2*V) + y, row<N, 0).to(tl.float32)
    g = tl.load(Z + row*(2*V) + V + y, row<N, 0).to(tl.float32)
    target, _, _, _ = _transform(z, g, A, S)
    tl.store(LSE+row, lse, row<N)
    tl.store(Loss+row, lse-target, row<N)


@tr.jit
def _gradient(Z, Y, LSE, N: tl.constexpr, V: tl.constexpr, TOTAL: tl.constexpr,
              A: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0, B)
    row, v = i//V, i%V
    z = tl.load(Z+row*(2*V)+v, row<N, 0).to(tl.float32)
    g = tl.load(Z+row*(2*V)+V+v, row<N, 0).to(tl.float32)
    logits, t, gate, cap = _transform(z,g,A,S)
    lse = tl.load(LSE+row, row<N, 0)
    y = tl.load(Y+row, row<N, 0)
    e = (tl.exp(logits-lse)-(v==y).to(tl.float32))/TOTAL
    d = e*(1.-cap*cap)
    tl.store(Z+row*(2*V)+v, d*gate, row<N)
    tl.store(Z+row*(2*V)+V+v, d*z*A*(1.-t*t), row<N)


def _compute(h, weight, targets, alpha, chunk_size, need_h, need_w, kernel):
    shape = h.shape
    h = h.reshape(-1, shape[-1]).contiguous()
    y = targets.reshape(-1).contiguous()
    n, d = h.shape
    v = weight.shape[0]//2
    w = weight.to(h.dtype).contiguous()
    need_grad = need_h or need_w
    dh = torch.empty_like(h) if need_h else None
    dw = torch.zeros_like(weight, dtype=torch.float32) if need_w else None
    losses = torch.empty(n, device=h.device, dtype=torch.float32)
    chunk = min(chunk_size, n)
    storage = torch.empty((chunk, 2*v), device=h.device, dtype=h.dtype)
    parts = tr.cdiv(v, 2048)
    partial = torch.empty((chunk, parts), device=h.device, dtype=torch.float32)
    lse = torch.empty(chunk, device=h.device, dtype=torch.float32)
    for start in range(0, n, chunk):
        end = min(start+chunk,n); m=end-start
        x, z = h[start:end], storage[:m]
        torch.mm(x, w.T, out=z)
        if kernel == 'row':
            _row_ce[(m,)](z,y[start:end],losses[start:end],m,v,n,alpha,15.,need_grad,
                          tr.next_power_of_2(v),num_warps=32 if v>=16384 else 8,enable_fp_fusion=False)
        else:
            _partial_lse[(m,parts)](z,partial,m,v,parts,alpha,15.,2048,num_warps=4,enable_fp_fusion=False)
            _finish_lse[(tr.cdiv(m,32),)](z,y[start:end],partial,lse,losses[start:end],m,v,parts,
                                         alpha,15.,tr.next_power_of_2(parts),32,num_warps=4,enable_fp_fusion=False)
            if need_grad:
                _gradient[(tr.cdiv(m*v,1024),)](z,y[start:end],lse,m,v,n,alpha,15.,1024,
                                               num_warps=4,enable_fp_fusion=False)
        if need_h:
            torch.mm(z,w,out=dh[start:end])
        if need_w:
            if h.dtype in (torch.bfloat16,torch.float16):
                torch.addmm(dw,z.T,x,out_dtype=torch.float32,out=dw)
            else:
                torch.addmm(dw,z.T,x,out=dw)
    # Match the original BF16 classifier's cast-back gradient boundary while
    # accumulating all token chunks in FP32 first.
    if dw is not None: dw = dw.to(h.dtype).to(weight.dtype)
    return losses.mean(), dh.reshape(shape) if dh is not None else None, dw


class _LinearGateCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,weight,targets,alpha,chunk_size,kernel):
        loss,dh,dw = _compute(h,weight,targets,alpha,chunk_size,*ctx.needs_input_grad[:2],kernel)
        ctx.save_for_backward(dh,dw)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        dh,dw = ctx.saved_tensors
        return (dh*grad if dh is not None else None,
                dw*grad if dw is not None else None,None,None,None,None)


def linear_gate_cross_entropy(h,weight,targets,alpha=.5,chunk_size=2048,kernel='tiled'):
    if not h.is_cuda or h.dtype not in (torch.bfloat16,torch.float32):
        raise ValueError('Vocabulary gate Triton backend requires CUDA BF16 or FP32')
    if weight.ndim!=2 or weight.shape[0]%2 or weight.shape[1]!=h.shape[-1] or targets.shape!=h.shape[:-1]:
        raise ValueError('Incompatible packed gate / target shapes')
    if not 0 <= alpha <= 1 or chunk_size < 1 or kernel not in ('row','tiled'):
        raise ValueError('Invalid vocabulary gate configuration')
    if torch.is_grad_enabled() and (h.requires_grad or weight.requires_grad):
        return _LinearGateCE.apply(h,weight,targets,alpha,chunk_size,kernel)
    return _compute(h,weight,targets,alpha,chunk_size,False,False,kernel)[0]
