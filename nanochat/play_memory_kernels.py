"""Fused play scan/ReLU² and analytical reverse scan, FP32 state/reductions.

Two forward kernels, three backward kernels for multiple time chunks.
No atomics, expanded gate bank, dense projection, or Python time loop.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _clamp(x, lo, hi):
    return tl.minimum(tl.maximum(x, lo), hi)


@tr.jit
def _interval(l1, h1, l2, h2):
    return _clamp(l1, l2, h2), _clamp(h1, l2, h2)


@tr.jit
def _affine(a1, b1, a2, b2):
    return a2 * a1, a2 * b1 + b2


@tr.jit
def _summary(U, R, SL, SH, T: tl.constexpr, H: tl.constexpr,
             N: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    p = tl.program_id(1)
    b, k = p // N, p % N
    i = tl.arange(0, BT)
    t = k * BT + i
    u = tl.load(U + (b*T + t[:, None])*H + c[None, :],
                (t[:, None] < T) & (c[None, :] < H), other=0).to(tl.float32)
    r = tl.load(R+c, c < H, other=0)
    lo = tl.where(t[:, None] < T, u-r[None, :], -float('inf'))
    hi = tl.where(t[:, None] < T, u+r[None, :], float('inf'))
    lo, hi = tl.associative_scan((lo, hi), 0, _interval)
    last_l = tl.max(tl.where(i[:, None] == BT-1, lo, -float('inf')), 0)
    last_h = tl.min(tl.where(i[:, None] == BT-1, hi, float('inf')), 0)
    tl.store(SL+p*H+c, last_l, c < H)
    tl.store(SH+p*H+c, last_h, c < H)


@tr.jit
def _forward(U, R, W, SL, SH, M, Y, T: tl.constexpr, H: tl.constexpr,
             N: tl.constexpr, NC: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    p = tl.program_id(1)
    b, k = p // N, p % N
    initial = tl.full((BC,), 0, tl.float32)
    if N > 1:
        j = tl.arange(0, NC)
        valid = (j[:, None] < k) & (c[None, :] < H)
        lo = tl.load(SL+(b*N+j[:, None])*H+c[None, :], valid, other=-float('inf'))
        hi = tl.load(SH+(b*N+j[:, None])*H+c[None, :], valid, other=float('inf'))
        lo, hi = tl.associative_scan((lo, hi), 0, _interval)
        last_l = tl.max(tl.where(j[:, None] == NC-1, lo, -float('inf')), 0)
        last_h = tl.min(tl.where(j[:, None] == NC-1, hi, float('inf')), 0)
        initial = _clamp(initial, last_l, last_h)
    i = tl.arange(0, BT)
    t = k*BT+i
    offsets = (b*T+t[:, None])*H+c[None, :]
    valid = (t[:, None] < T) & (c[None, :] < H)
    u = tl.load(U+offsets, valid, other=0).to(tl.float32)
    r = tl.load(R+c, c < H, other=0)
    w = tl.load(W+c, c < H, other=0)
    lo, hi = tl.associative_scan((u-r[None, :], u+r[None, :]), 0, _interval)
    m = _clamp(initial[None, :], lo, hi)
    z = tl.maximum(u+w[None, :]*m, 0)
    tl.store(M+offsets, m, valid)
    tl.store(Y+offsets, z*z, valid)


@tr.jit
def _back_values(U, R, W, M, G, b, k, c, T: tl.constexpr, H: tl.constexpr,
                 BT: tl.constexpr):
    i = tl.arange(0, BT)
    t = k*BT+BT-1-i
    offsets = (b*T+t[:, None])*H+c[None, :]
    valid = (t[:, None] < T) & (c[None, :] < H)
    u = tl.load(U+offsets, valid, other=0).to(tl.float32)
    m = tl.load(M+offsets, valid, other=0)
    g = tl.load(G+offsets, valid, other=0).to(tl.float32)
    r = tl.load(R+c, c < H, other=0)
    w = tl.load(W+c, c < H, other=0)
    next_u = tl.load(U+offsets+H, (t[:, None]+1 < T) & (c[None, :] < H), other=0).to(tl.float32)
    retained_next = (m >= next_u-r[None, :]) & (m <= next_u+r[None, :]) & (t[:, None]+1 < T)
    dz = 2*tl.maximum(u+w[None, :]*m, 0)*g
    # dstate[t] = dz[t]*w + retained[t+1]*dstate[t+1].
    a = tl.where(t[:, None] < T, retained_next.to(tl.float32), 1.)
    q = tl.where(t[:, None] < T, dz*w[None, :], 0.)
    return t, offsets, valid, u, m, r, dz, a, q


@tr.jit
def _back_summary(U, R, W, M, G, SA, SB, T: tl.constexpr, H: tl.constexpr,
                  N: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    c = tl.program_id(0)*BC+tl.arange(0, BC)
    p = tl.program_id(1)
    t, offsets, valid, u, m, r, dz, a, q = _back_values(U,R,W,M,G,p//N,p%N,c,T,H,BT)
    a, q = tl.associative_scan((a, q), 0, _affine)
    last = tl.arange(0, BT)[:, None] == BT-1
    tl.store(SA+p*H+c, tl.sum(tl.where(last,a,0),0), c < H)
    tl.store(SB+p*H+c, tl.sum(tl.where(last,q,0),0), c < H)


@tr.jit
def _backward(U, R, W, M, G, SA, SB, DU, PR, PW, T: tl.constexpr,
              H: tl.constexpr, N: tl.constexpr, NC: tl.constexpr,
              BT: tl.constexpr, BC: tl.constexpr):
    c = tl.program_id(0)*BC+tl.arange(0, BC)
    p = tl.program_id(1)
    b, k = p//N, p%N
    carry = tl.full((BC,),0,tl.float32)
    if N > 1:
        i = tl.arange(0, NC)
        j = N-1-i
        valid = (j[:, None] > k) & (c[None, :] < H)
        a = tl.load(SA+(b*N+j[:, None])*H+c[None, :], valid, other=1.)
        q = tl.load(SB+(b*N+j[:, None])*H+c[None, :], valid, other=0.)
        a, q = tl.associative_scan((a,q),0,_affine)
        carry = tl.sum(tl.where(i[:, None] == NC-1,q,0),0)
    t, offsets, valid, u, m, r, dz, a, q = _back_values(U,R,W,M,G,b,k,c,T,H,BT)
    a, q = tl.associative_scan((a,q),0,_affine)
    dm = a*carry[None, :]+q
    previous = tl.load(M+offsets-H, valid & (t[:, None] > 0), other=0)
    low = previous < u-r[None, :]
    high = previous > u+r[None, :]
    du = dz+tl.where(low | high,dm,0)
    dr = tl.where(low,-dm,tl.where(high,dm,0))
    tl.store(DU+offsets,du,valid)
    tl.store(PR+p*H+c,tl.sum(tl.where(valid,dr,0),0),c < H)
    tl.store(PW+p*H+c,tl.sum(tl.where(valid,dz*m,0),0),c < H)


@tr.jit
def _parameter_reduce(PR, PW, DR, DW, P: tl.constexpr, H: tl.constexpr,
                      BP: tl.constexpr, BC: tl.constexpr):
    c = tl.program_id(0)*BC+tl.arange(0, BC)
    p = tl.arange(0, BP)
    valid = (p[:, None] < P) & (c[None, :] < H)
    dr = tl.load(PR+p[:, None]*H+c[None, :],valid,other=0)
    dw = tl.load(PW+p[:, None]*H+c[None, :],valid,other=0)
    tl.store(DR+c,tl.sum(dr,0),c < H)
    tl.store(DW+c,tl.sum(dw,0),c < H)


class _Play(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, radius, weight):
        assert u.is_cuda and u.ndim == 3 and u.is_contiguous()
        assert u.dtype in (torch.bfloat16, torch.float32)
        assert radius.dtype == weight.dtype == torch.float32
        b,t,h = u.shape
        assert 0 < t <= 8192 and radius.shape == weight.shape == (h,)
        bt,bc = 128,32
        n = tr.cdiv(t,bt)
        sl = torch.empty((b*n,h),device=u.device,dtype=torch.float32)
        sh = torch.empty_like(sl)
        m = torch.empty_like(u,dtype=torch.float32)
        y = torch.empty_like(u)
        grid = (tr.cdiv(h,bc),b*n)
        if n > 1:
            _summary[grid](u,radius,sl,sh,t,h,n,bt,bc,enable_fp_fusion=False)
        _forward[grid](u,radius,weight,sl,sh,m,y,t,h,n,tr.next_power_of_2(n),bt,bc,enable_fp_fusion=False)
        ctx.save_for_backward(u,radius,weight,m)
        return y

    @staticmethod
    def backward(ctx, grad):
        u,radius,weight,m = ctx.saved_tensors
        grad = grad.contiguous()
        b,t,h = u.shape
        bt,bc = 128,32
        n = tr.cdiv(t,bt)
        sa = torch.empty((b*n,h),device=u.device,dtype=torch.float32)
        sb,pr,pw = torch.empty_like(sa),torch.empty_like(sa),torch.empty_like(sa)
        du,dr,dw = torch.empty_like(u),torch.empty_like(radius),torch.empty_like(weight)
        grid = (tr.cdiv(h,bc),b*n)
        if n > 1:
            _back_summary[grid](u,radius,weight,m,grad,sa,sb,t,h,n,bt,bc,enable_fp_fusion=False)
        _backward[grid](u,radius,weight,m,grad,sa,sb,du,pr,pw,t,h,n,tr.next_power_of_2(n),bt,bc,enable_fp_fusion=False)
        _parameter_reduce[(tr.cdiv(h,bc),)](pr,pw,dr,dw,b*n,h,tr.next_power_of_2(b*n),bc)
        return du,dr,dw


def play_relu2(u, radius, weight):
    return _Play.apply(u, radius, weight)
