"""HCA training kernels for separate K/V (Hopper, FP32/BF16).

Independent Triton implementation of the reference in hca.py. The execution
design follows NVIDIA's fused Compressor (chunk ownership, recomputation in
backward), not its MLA layout or ratio restrictions. Fixed and Dynamic keep
the reference's visible open prefix. Dynamic backward retains zero-gate STE
derivatives; forward compaction alone would not preserve that estimator.

No T x T intermediates, no atomic gradient updates, no Python token loop.
Dynamic probabilities and score adjoints remain FP32 in dot products because
BF16 rounding can dominate a small, cancelling router-bias gradient. Fixed
BF16 attention uses BF16 matrix products with FP32 accumulators, following the
official DeepSeek attention precision scheme; FP32 oracle inputs stay precise.
GPU validation is mandatory before selecting this backend for a formal run.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice
from torch.autograd.function import once_differentiable


@tr.jit
def _affine(a, m, k, v, aa, mm, kk, vv):
    return aa * a, mm + aa * m, kk + aa * k, vv + aa * v


@tr.jit
def _pool_fwd(K, V, S, G, M, SK, SV, PK, PV,
              T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
              R: tl.constexpr, FIXED: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    rows = tl.arange(0, BT)
    cm = tl.full((BD,), 0, tl.float32)
    ck = tl.full((BD,), 0, tl.float32)
    cv = tl.full((BD,), 0, tl.float32)
    if FIXED:
        lo = tl.program_id(2) * R
        hi = lo + R
    else:
        lo, hi = 0, T
    for base in range(lo, hi, BT):
        t = base + rows
        valid = (t[:, None] < T) & (t[:, None] < hi) & (d[None, :] < D)
        off = ((b * T + t[:, None]) * H + h) * D + d[None, :]
        s = tl.load(S + off, valid, 0).to(tl.float32)
        u = libdevice.tanh(s / 10)
        w = tl.exp(10 * u)
        k = tl.load(K + off, valid, 0).to(tl.float32)
        v = tl.load(V + off, valid, 0).to(tl.float32)
        if FIXED:
            m = tl.cumsum(w, 0)
            sk = tl.cumsum(w * k, 0)
            sv = tl.cumsum(w * v, 0)
        else:
            a = 1 - tl.load(G + b * T + t, t < T, 1)
            aa, m, sk, sv = tl.associative_scan(
                (tl.broadcast_to(a[:, None], (BT, BD)), w, w*k, w*v), 0, _affine)
            m += aa * cm[None, :]
            sk += aa * ck[None, :]
            sv += aa * cv[None, :]
            cm = tl.sum(tl.where(rows[:, None] == BT-1, m, 0), 0)
            ck = tl.sum(tl.where(rows[:, None] == BT-1, sk, 0), 0)
            cv = tl.sum(tl.where(rows[:, None] == BT-1, sv, 0), 0)
        if not FIXED:
            tl.store(M + off, m, valid)
            tl.store(SK + off, sk, valid)
            tl.store(SV + off, sv, valid)
        tl.store(PK + off, sk / m, valid)
        tl.store(PV + off, sv / m, valid)


@tr.jit
def _pool_bwd(K, V, S, G, M, SK, SV, DPK, DPV, DK, DV, DS, DG,
              T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, ND: tl.constexpr,
              R: tl.constexpr, FIXED: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    dc = tl.program_id(1)
    d = dc * BD + tl.arange(0, BD)
    rows = tl.arange(0, BT)
    am = tl.full((BD,), 0, tl.float32)
    ak = tl.full((BD,), 0, tl.float32)
    av = tl.full((BD,), 0, tl.float32)
    if FIXED:
        lo = tl.program_id(2) * R
        hi = lo + BT
    else:
        lo, hi = 0, tl.cdiv(T, BT) * BT
    for step in range(lo, hi, BT):
        base = hi - BT - (step - lo)
        t = base + rows
        valid = (t[:, None] < T) & (d[None, :] < D)
        if FIXED:
            valid = valid & (t[:, None] < lo + R)
        off = ((b * T + t[:, None]) * H + h) * D + d[None, :]
        # Independent fixed chunks can recompute their prefix sums in registers.
        # This avoids three full FP32 state writes/reads and saved activations.
        s = tl.load(S + off, valid, 0).to(tl.float32)
        u = libdevice.tanh(s / 10)
        w = tl.exp(10 * u)
        k = tl.load(K + off, valid, 0).to(tl.float32)
        v = tl.load(V + off, valid, 0).to(tl.float32)
        if FIXED:
            m = tl.cumsum(w, 0)
            sk = tl.cumsum(w*k, 0)
            sv = tl.cumsum(w*v, 0)
        else:
            m = tl.load(M + off, valid, 1)
            sk = tl.load(SK + off, valid, 0)
            sv = tl.load(SV + off, valid, 0)
        gk = tl.load(DPK + off, valid, 0).to(tl.float32)
        gv = tl.load(DPV + off, valid, 0).to(tl.float32)
        dk, dv = gk / m, gv / m
        dm = -(dk * sk + dv * sv) / m
        if FIXED:
            dm = tl.cumsum(dm, 0, reverse=True)
            dk = tl.cumsum(dk, 0, reverse=True)
            dv = tl.cumsum(dv, 0, reverse=True)
        else:
            a = 1 - tl.load(G + b*T+t+1, t+1 < T, 1)
            aa, dm, dk, dv = tl.associative_scan(
                (tl.broadcast_to(a[:, None], (BT, BD)), dm, dk, dv),
                0, _affine, reverse=True)
            dm += aa * am[None, :]
            dk += aa * ak[None, :]
            dv += aa * av[None, :]
            am = tl.sum(tl.where(rows[:, None] == 0, dm, 0), 0)
            ak = tl.sum(tl.where(rows[:, None] == 0, dk, 0), 0)
            av = tl.sum(tl.where(rows[:, None] == 0, dv, 0), 0)
            prior = valid & (t[:, None] > 0)
            pm = tl.load(M + off - H*D, prior, 0)
            psk = tl.load(SK + off - H*D, prior, 0)
            psv = tl.load(SV + off - H*D, prior, 0)
            dg = -tl.sum(tl.where(d[None, :] < D, dm*pm + dk*psk + dv*psv, 0), 1)
            tl.store(DG + ((b*T+t)*H+h)*ND+dc, dg, t < T)
        tl.store(DK + off, dk*w, valid)
        tl.store(DV + off, dv*w, valid)
        tl.store(DS + off, w*(dm + dk*k + dv*v)*(1-u*u), valid)


class _Pool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, score, starts, ratio):
        k, v, score, starts = [x.contiguous() for x in (k, v, score, starts)]
        b, t, h, d = k.shape
        fixed = ratio > 0
        bt = tr.next_power_of_2(ratio) if fixed else 64
        bd = 32
        # Fixed kernels never dereference the dummy state pointers.
        m, sk, sv = (k, k, k) if fixed else [torch.empty_like(k, dtype=torch.float32) for _ in range(3)]
        pk, pv = torch.empty_like(k), torch.empty_like(v)
        grid = (b*h, tr.cdiv(d, bd), tr.cdiv(t, ratio) if fixed else 1)
        _pool_fwd[grid](k, v, score, starts, m, sk, sv, pk, pv,
                       t, h, d, ratio, fixed, bt, bd, enable_fp_fusion=False)
        ctx.save_for_backward(k, v, score, starts, m, sk, sv)
        ctx.params = ratio, bt, bd, grid
        return pk, pv

    @staticmethod
    @once_differentiable
    def backward(ctx, gk, gv):
        k, v, score, starts, m, sk, sv = ctx.saved_tensors
        ratio, bt, bd, grid = ctx.params
        b, t, h, d = k.shape
        dk, dv, ds = [torch.empty_like(x) for x in (k, v, score)]
        dg = torch.empty((b, t, h, tr.cdiv(d, bd)), device=k.device, dtype=torch.float32)
        _pool_bwd[grid](k, v, score, starts, m, sk, sv, gk.contiguous(), gv.contiguous(),
                       dk, dv, ds, dg, t, h, d, tr.cdiv(d, bd), ratio, ratio > 0,
                       bt, bd, enable_fp_fusion=False)
        return dk, dv, ds, None if ratio > 0 else dg.sum((2, 3)), None


def prefix_pool(k, v, score, starts, ratio=0):
    """ratio=0: arbitrary hard STE starts; ratio>0: independent fixed chunks."""
    if not k.is_cuda or k.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('HCA Triton pooling requires CUDA FP32/BF16')
    if k.shape != v.shape or k.shape != score.shape or starts.shape != k.shape[:2]:
        raise ValueError('HCA pooling shapes do not match')
    if ratio < 0 or ratio > 128:
        raise ValueError('Fixed Triton pooling supports block sizes 1..128')
    return _Pool.apply(k, v, score, starts, ratio)


@tr.jit
def _pack(G, INDEX, COUNTS, T: tl.constexpr, BT: tl.constexpr):
    b = tl.program_id(0)
    j = tl.arange(0, BT)
    closed = tl.load(G+b*T+j+1, j+1 < T, 0).to(tl.int32)
    count = tl.cumsum(closed)
    tl.store(INDEX+b*T+count-1, j, (j < T) & (closed != 0))
    tl.store(COUNTS+b*T+j, count, j < T)


@tr.jit
def _attn_fwd(Q, K, V, PK, PV, INDEX, COUNTS, O, OF, LSE,
              T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
              R: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    rows, cols, d = tl.arange(0, BM), tl.arange(0, BN), tl.arange(0, BD)
    t = block * BM + rows
    off = ((b*T+t[:, None])*H+h)*D+d[None, :]
    valid = (t[:, None] < T) & (d[None, :] < D)
    q = tl.load(Q+off, valid, 0)
    pk = tl.load(PK+off, valid, 0).to(tl.float32)
    pv = tl.load(PV+off, valid, 0).to(tl.float32)
    scale: tl.constexpr = D ** -.5
    # Every row has its own causal open-prefix summary. Initialize the common
    # online normalization with it, then fold local and completed entries.
    m = tl.sum(q.to(tl.float32)*pk, 1)*scale
    l = tl.full((BM,), 1, tl.float32)
    acc = pv
    for base in range(tl.maximum(0, block*BM-W+1), tl.minimum(T, (block+1)*BM), BN):
        j = base+cols
        ko = ((b*T+j[:, None])*H+h)*D+d[None, :]
        kv = (j[:, None] < T) & (d[None, :] < D)
        k = tl.load(K+ko, kv, 0)
        v = tl.load(V+ko, kv, 0)
        z = tl.dot(q, tl.trans(k), input_precision='ieee')*scale
        allowed = (j[None, :] <= t[:, None]) & (t[:, None]-j[None, :] < W) & (j[None, :] < T)
        z = tl.where(allowed, z, -float('inf'))
        new_m = tl.maximum(m, tl.max(z, 1))
        p = tl.exp(z-new_m[:, None])
        alpha = tl.exp(m-new_m)
        if R > 0 and q.dtype == tl.bfloat16:
            acc = acc*alpha[:, None]+tl.dot(p.to(tl.bfloat16), v)
        else:
            acc = acc*alpha[:, None]+tl.dot(p, v.to(tl.float32), input_precision='tf32x3')
        l = l*alpha+tl.sum(p, 1)
        m = new_m
    last = tl.minimum(T-1, (block+1)*BM-1)
    if R > 0:
        n = last // R
    else:
        n = tl.load(COUNTS+b*T+last-1, last > 0, 0)
    for base in range(0, n, BN):
        c = base+cols
        if R > 0:
            j = (c+1)*R-1
        else:
            j = tl.load(INDEX+b*T+c, c < n, 0)
        ko = ((b*T+j[:, None])*H+h)*D+d[None, :]
        kv = (c[:, None] < n) & (d[None, :] < D)
        k = tl.load(PK+ko, kv, 0)
        v = tl.load(PV+ko, kv, 0)
        z = tl.dot(q, tl.trans(k), input_precision='ieee')*scale
        z = tl.where((c[None, :] < n) & (j[None, :] < t[:, None]), z, -float('inf'))
        new_m = tl.maximum(m, tl.max(z, 1))
        p = tl.exp(z-new_m[:, None])
        alpha = tl.exp(m-new_m)
        if R > 0 and q.dtype == tl.bfloat16:
            acc = acc*alpha[:, None]+tl.dot(p.to(tl.bfloat16), v)
        else:
            acc = acc*alpha[:, None]+tl.dot(p, v.to(tl.float32), input_precision='tf32x3')
        l = l*alpha+tl.sum(p, 1)
        m = new_m
    out = acc/l[:, None]
    tl.store(O+off, out, valid)
    tl.store(OF+off, out, valid)
    tl.store(LSE+bh*T+t, m+tl.log(l), t < T)


@tr.jit
def _attn_dq(Q, K, V, PK, PV, INDEX, COUNTS, DO, OF, LSE, DQ,
             T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
             R: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    t = block*BM+tl.arange(0, BM)
    cols, d = tl.arange(0, BN), tl.arange(0, BD)
    off = ((b*T+t[:, None])*H+h)*D+d[None, :]
    valid = (t[:, None] < T) & (d[None, :] < D)
    q = tl.load(Q+off, valid, 0)
    do = tl.load(DO+off, valid, 0)
    o = tl.load(OF+off, valid, 0)
    lse = tl.load(LSE+bh*T+t, t < T, 0)
    delta = tl.sum(do.to(tl.float32)*o, 1)
    diag_k = tl.load(PK+off, valid, 0).to(tl.float32)
    diag_v = tl.load(PV+off, valid, 0).to(tl.float32)
    scale: tl.constexpr = D ** -.5
    diag_p = tl.exp(tl.sum(q.to(tl.float32)*diag_k, 1)*scale-lse)
    diag_ds = diag_p*(tl.sum(do.to(tl.float32)*diag_v, 1)-delta)
    dq = diag_ds[:, None]*diag_k*scale
    for base in range(tl.maximum(0, block*BM-W+1), tl.minimum(T, (block+1)*BM), BN):
        j = base+cols
        ko = ((b*T+j[:, None])*H+h)*D+d[None, :]
        kv = (j[:, None] < T) & (d[None, :] < D)
        k = tl.load(K+ko, kv, 0)
        v = tl.load(V+ko, kv, 0)
        z = tl.dot(q, tl.trans(k), input_precision='ieee')*scale
        allowed = (j[None, :] <= t[:, None]) & (t[:, None]-j[None, :] < W) & (j[None, :] < T)
        p = tl.exp(tl.where(allowed, z-lse[:, None], -float('inf')))
        dp = tl.dot(do, tl.trans(v), input_precision='ieee')
        ds = p*(dp-delta[:, None])
        if R > 0 and q.dtype == tl.bfloat16:
            dq += tl.dot(ds.to(tl.bfloat16), k)*scale
        else:
            dq += tl.dot(ds, k.to(tl.float32), input_precision='tf32x3')*scale
    last = tl.minimum(T-1, (block+1)*BM-1)
    if R > 0:
        n = last // R
    else:
        n = tl.load(COUNTS+b*T+last-1, last > 0, 0)
    for base in range(0, n, BN):
        c = base+cols
        if R > 0:
            j = (c+1)*R-1
        else:
            j = tl.load(INDEX+b*T+c, c < n, 0)
        ko = ((b*T+j[:, None])*H+h)*D+d[None, :]
        kv = (c[:, None] < n) & (d[None, :] < D)
        k = tl.load(PK+ko, kv, 0)
        v = tl.load(PV+ko, kv, 0)
        z = tl.dot(q, tl.trans(k), input_precision='ieee')*scale
        p = tl.exp(tl.where((c[None, :] < n) & (j[None, :] < t[:, None]), z-lse[:, None], -float('inf')))
        dp = tl.dot(do, tl.trans(v), input_precision='ieee')
        ds = p*(dp-delta[:, None])
        if R > 0 and q.dtype == tl.bfloat16:
            dq += tl.dot(ds.to(tl.bfloat16), k)*scale
        else:
            dq += tl.dot(ds, k.to(tl.float32), input_precision='tf32x3')*scale
    tl.store(DQ+off, dq, valid)


@tr.jit
def _prefix_diag_bwd(Q, PK, PV, DO, OF, LSE, DK, DV,
                     T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                     BM: tl.constexpr, BD: tl.constexpr):
    bh = tl.program_id(1)
    b, h = bh//H, bh%H
    t = tl.program_id(0)*BM+tl.arange(0, BM)
    d = tl.arange(0, BD)
    off = ((b*T+t[:, None])*H+h)*D+d[None, :]
    valid = (t[:, None] < T) & (d[None, :] < D)
    q = tl.load(Q+off, valid, 0).to(tl.float32)
    k = tl.load(PK+off, valid, 0).to(tl.float32)
    v = tl.load(PV+off, valid, 0).to(tl.float32)
    do = tl.load(DO+off, valid, 0).to(tl.float32)
    out = tl.load(OF+off, valid, 0)
    lse = tl.load(LSE+bh*T+t, t < T, 0)
    p = tl.exp(tl.sum(q*k, 1)*(D**-.5)-lse)
    ds = p*tl.sum(do*(v-out), 1)
    tl.store(DK+off, ds[:, None]*q*(D**-.5), valid)
    tl.store(DV+off, p[:, None]*do, valid)


@tr.jit
def _attn_dkv(Q, K, V, G, DO, OF, LSE, DK, DV, DG,
              T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
              R: tl.constexpr, SOURCE: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    # SOURCE 0: local; 1: dynamic prefix incl. zero-gate counterfactuals;
    # 2: fixed completed summaries (diagonal gradients already written).
    block, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh//H, bh%H
    c = block*BN+tl.arange(0, BN)
    d = tl.arange(0, BD)
    if SOURCE == 2:
        j = (c+1)*R-1
        jvalid = j < T-1
        first = (block*BN+1)*R-1
    else:
        j = c
        jvalid = j < T
        first = block*BN
    off = ((b*T+j[:, None])*H+h)*D+d[None, :]
    valid = jvalid[:, None] & (d[None, :] < D)
    k = tl.load(K+off, valid, 0)
    v = tl.load(V+off, valid, 0)
    dk = tl.zeros((BN, BD), tl.float32)
    dv = tl.zeros((BN, BD), tl.float32)
    dg = tl.zeros((BN,), tl.float32)
    if SOURCE == 1:
        closed = tl.load(G+b*T+j+1, j+1 < T, 0)
    hi = T
    if SOURCE == 0:
        hi = tl.minimum(T, (block+1)*BN+W-1)
    for base in range(first//BM*BM, hi, BM):
        t = base+tl.arange(0, BM)
        qo = ((b*T+t[:, None])*H+h)*D+d[None, :]
        qvalid = (t[:, None] < T) & (d[None, :] < D)
        q = tl.load(Q+qo, qvalid, 0)
        do = tl.load(DO+qo, qvalid, 0)
        out = tl.load(OF+qo, qvalid, 0)
        lse = tl.load(LSE+bh*T+t, t < T, 0)
        delta = tl.sum(do.to(tl.float32)*out, 1)
        z = tl.dot(q, tl.trans(k), input_precision='ieee')*(D**-.5)
        allowed = (t[:, None] < T) & jvalid[None, :] & (t[:, None] >= j[None, :])
        if SOURCE == 0:
            allowed = allowed & (t[:, None]-j[None, :] < W)
        if SOURCE == 2:
            allowed = allowed & (t[:, None] > j[None, :])
        e = tl.exp(tl.where(allowed, z-lse[:, None], -float('inf')))
        dp = tl.dot(do, tl.trans(v), input_precision='ieee')-delta[:, None]
        if SOURCE == 1:
            # This derivative must also be evaluated when closed[j] == 0.
            dg += tl.sum(tl.where(t[:, None] > j[None, :], e*dp, 0), 0)
            p = e*tl.where(t[:, None] == j[None, :], 1., closed[None, :])
        else:
            p = e
        ds = p*dp
        if R > 0 and q.dtype == tl.bfloat16:
            dk += tl.dot(tl.trans(ds.to(tl.bfloat16)), q)*(D**-.5)
            dv += tl.dot(tl.trans(p.to(tl.bfloat16)), do)
        else:
            dk += tl.dot(tl.trans(ds), q.to(tl.float32), input_precision='tf32x3')*(D**-.5)
            dv += tl.dot(tl.trans(p), do.to(tl.float32), input_precision='tf32x3')
    if SOURCE == 2:
        dk += tl.load(DK+off, valid, 0)
        dv += tl.load(DV+off, valid, 0)
    tl.store(DK+off, dk, valid)
    tl.store(DV+off, dv, valid)
    if SOURCE == 1:
        # gate j is start[j+1]; start[0] is forced and gets no visibility grad.
        tl.store(DG+(b*T+j+1)*H+h, dg, j+1 < T)
        if block == 0:
            tl.store(DG+b*T*H+h, 0.)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pk, pv, starts, window, ratio):
        q, k, v, pk, pv, starts = [x.contiguous() for x in (q, k, v, pk, pv, starts)]
        b, t, h, d = q.shape
        index = torch.empty((b, t), device=q.device, dtype=torch.int32)
        counts = torch.empty_like(index)
        if ratio == 0:
            _pack[(b,)](starts, index, counts, t, tr.next_power_of_2(t))
        out = torch.empty_like(q)
        of = torch.empty_like(q, dtype=torch.float32)
        lse = torch.empty((b, h, t), device=q.device, dtype=torch.float32)
        _attn_fwd[(tr.cdiv(t, 32), b*h)](q, k, v, pk, pv, index, counts, out, of, lse,
                        t, h, d, window, ratio, 32, 32, tr.next_power_of_2(d))
        ctx.save_for_backward(q, k, v, pk, pv, starts, index, counts, of, lse)
        ctx.params = window, ratio
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        q, k, v, pk, pv, starts, index, counts, of, lse = ctx.saved_tensors
        w, r = ctx.params
        b, t, h, d = q.shape
        do = do.contiguous()
        dq, dk, dv = [torch.empty_like(x) for x in (q, k, v)]
        # FP32 prefix buffers avoid rounding once per source before adding.
        dpk, dpv = [torch.empty_like(x, dtype=torch.float32) for x in (pk, pv)]
        dg = torch.empty((b, t, h), device=q.device, dtype=torch.float32)
        grid, bd = (tr.cdiv(t, 32), b*h), tr.next_power_of_2(d)
        _attn_dq[grid](q, k, v, pk, pv, index, counts, do, of, lse, dq, t, h, d, w, r, 32, 32, bd)
        _attn_dkv[grid](q, k, v, starts, do, of, lse, dk, dv, dg, t, h, d, w, r, 0, 32, 32, bd)
        if r > 0:
            _prefix_diag_bwd[grid](q, pk, pv, do, of, lse, dpk, dpv, t, h, d, 32, bd)
            n = (t-1)//r
            if n:
                _attn_dkv[(tr.cdiv(n, 32), b*h)](q, pk, pv, starts, do, of, lse,
                        dpk, dpv, dg, t, h, d, w, r, 2, 32, 32, bd)
        else:
            _attn_dkv[grid](q, pk, pv, starts, do, of, lse, dpk, dpv, dg,
                           t, h, d, w, r, 1, 32, 32, bd)
        return dq, dk, dv, dpk.to(pk.dtype), dpv.to(pv.dtype), None if r else dg.sum(2), None, None


def attention(q, k, v, pk, pv, starts, window, ratio=0):
    if not q.is_cuda or q.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('HCA Triton attention requires CUDA FP32/BF16')
    if any(x.shape != q.shape or x.dtype != q.dtype for x in (k, v, pk, pv)):
        raise ValueError('HCA attention requires matching separate Q/K/V shapes and dtypes')
    if not 16 <= q.shape[-1] <= 128 or q.shape[1] > 8192:
        raise ValueError('Initial HCA Triton envelope: head_dim 16..128, sequence <=8192')
    return _Attention.apply(q, k, v, pk, pv, starts, window, ratio)
