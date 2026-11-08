# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Four-stream split/FMA mixing adapted from TE 2.20.2; license and source
# attribution: research/connections/scaled-hc-nvidia/LICENSE.NVIDIA, source.json.
"""Fused per-stream RMS / elementwise sigmoid / HC read or write and VJPs.

Each CTA owns complete channel rows so the RMS derivative is included in the
same backward scan. Gates are recomputed, never materialized as a state bank.
Parameter gradients use deterministic token partials and the existing reducer.
"""
import triton as tr
import triton.language as tl


@tr.jit
def _mix4(x, h):
    # NVIDIA's split/FMA order, including the corresponding stream permutation.
    m: tl.constexpr = x.shape[0]
    c: tl.constexpr = x.shape[1]
    x01, x23 = tl.split(x.reshape(m, c, 2, 2))
    x0, x1 = tl.split(x01)
    x2, x3 = tl.split(x23)
    h01, h23 = tl.split(tl.trans(h, (0, 2, 1)).reshape(m, 4, 2, 2))
    h0, h1 = tl.split(h01)
    h2, h3 = tl.split(h23)
    out = tl.full((m, c, 4), 0., tl.float32)
    out = tl.fma(x0[:, :, None], h0[:, None, :], out)
    out = tl.fma(x1[:, :, None], h1[:, None, :], out)
    out = tl.fma(x2[:, :, None], h2[:, None, :], out)
    return tl.fma(x3[:, :, None], h3[:, None, :], out)


@tr.jit
def _gate(x, a, b, C: tl.constexpr, EPS: tl.constexpr):
    inv = tl.rsqrt(tl.sum(x*x, 1)/C + EPS)
    u = x*inv[:, None, :]
    g = 2.*tl.sigmoid(tl.fma(u, a[None, :, :], b[None, :, :]))
    return g, u, inv


@tr.jit
def _gate_vjp(x, df, a, g, u, inv, C: tl.constexpr):
    dz = df*x*g*(1.-.5*g)
    du = dz*a[None, :, :]
    dx = df*g + inv[:, None, :]*(du-u*(tl.sum(du*u, 1)/C)[:, None, :])
    return dx, tl.sum(dz*u, 0), tl.sum(dz, 0)


_FWD = [tr.Config(dict(BM=m), num_warps=w, num_stages=1)
        for m in (1, 2, 4) for w in (4, 8)]
_BWD = [tr.Config(dict(BM=4), num_warps=w, num_stages=1) for w in (4, 8)]


@tr.autotune(configs=_FWD, key=['M', 'C', 'WRITE'])
@tr.jit
def dynamic_hc_fwd(X, Y, P, H, A, B, PA, PB, O,
                   M: tl.constexpr, C: tl.constexpr, WRITE: tl.constexpr,
                   EPS: tl.constexpr, BC: tl.constexpr, BM: tl.constexpr):
    t = tl.program_id(0)*BM + tl.arange(0, BM)
    c = tl.arange(0, BC)
    s = tl.arange(0, 4)
    mask = (t[:, None, None] < M) & (c[None, :, None] < C)
    off = t[:, None, None]*C*4+c[None, :, None]*4+s[None, None, :]
    x = tl.load(X+off, mask, 0).to(tl.float32)
    poff = c[:, None]*4+s[None, :]
    a = tl.load(A+poff, c[:, None] < C, 0)
    b = tl.load(B+poff, c[:, None] < C, 0)
    g, u, inv = _gate(x, a, b, C, EPS)
    xg = x*g
    p = tl.load(P+t[:, None]*4+s[None, :], t[:, None] < M, 0)
    if WRITE:
        j = tl.arange(0, 4)
        h = tl.load(H+t[:, None, None]*16+s[None, :, None]*4+j[None, None, :],
                    t[:, None, None] < M, 0)
        y = tl.load(Y+t[:, None]*C+c[None, :],
                    (t[:, None] < M) & (c[None, :] < C), 0).to(tl.float32)
        yi = tl.rsqrt(tl.sum(y*y, 1)/C+EPS)
        yu = y*yi[:, None]
        pa = tl.load(PA+poff, c[:, None] < C, 0)
        pb = tl.load(PB+poff, c[:, None] < C, 0)
        pg = 2.*tl.sigmoid(tl.fma(yu[:, :, None], pa[None, :, :], pb[None, :, :]))
        out = _mix4(xg, h)
        out = tl.fma(y[:, :, None]*pg, p[:, None, :], out)
        tl.store(O+off, out, mask)
    else:
        x01, x23 = tl.split(xg.reshape(BM, BC, 2, 2))
        x0, x1 = tl.split(x01)
        x2, x3 = tl.split(x23)
        p01, p23 = tl.split(p.reshape(BM, 2, 2))
        p0, p1 = tl.split(p01)
        p2, p3 = tl.split(p23)
        out = tl.full((BM, BC), 0., tl.float32)
        out = tl.fma(x0, p0[:, None], out)
        out = tl.fma(x1, p1[:, None], out)
        out = tl.fma(x2, p2[:, None], out)
        out = tl.fma(x3, p3[:, None], out)
        tl.store(O+t[:, None]*C+c[None, :], out,
                 (t[:, None] < M) & (c[None, :] < C))


@tr.autotune(configs=_BWD, key=['M', 'C', 'WRITE'])
@tr.jit
def dynamic_hc_bwd(X, Y, P, H, A, B, PA, PB, GO, GX, GY, GP, GH, PART,
                   M: tl.constexpr, C: tl.constexpr, WRITE: tl.constexpr,
                   EPS: tl.constexpr, BC: tl.constexpr, BM: tl.constexpr):
    pid = tl.program_id(0)
    t = pid*BM+tl.arange(0, BM)
    c = tl.arange(0, BC)
    s = tl.arange(0, 4)
    mask = (t[:, None, None] < M) & (c[None, :, None] < C)
    off = t[:, None, None]*C*4+c[None, :, None]*4+s[None, None, :]
    poff = c[:, None]*4+s[None, :]
    x = tl.load(X+off, mask, 0).to(tl.float32)
    a = tl.load(A+poff, c[:, None] < C, 0)
    b = tl.load(B+poff, c[:, None] < C, 0)
    g, u, inv = _gate(x, a, b, C, EPS)
    xg = x*g
    p = tl.load(P+t[:, None]*4+s[None, :], t[:, None] < M, 0)
    splits: tl.constexpr = tl.cdiv(M, 4)
    if WRITE:
        go = tl.load(GO+off, mask, 0).to(tl.float32)
        j = tl.arange(0, 4)
        h = tl.load(H+t[:, None, None]*16+s[None, :, None]*4+j[None, None, :],
                    t[:, None, None] < M, 0)
        df = _mix4(go, tl.trans(h, (0, 2, 1)))
        # IEEE FP32 channel reductions; no low-precision routing-gradient GEMM.
        for i in tl.static_range(4):
            xi = tl.sum(tl.where(s[None, None, :] == i, xg, 0.), 2)
            gh = tl.sum(go*xi[:, :, None], 1)
            tl.store(GH+t[:, None]*16+i*4+s[None, :], gh, t[:, None] < M)
        y = tl.load(Y+t[:, None]*C+c[None, :],
                    (t[:, None] < M) & (c[None, :] < C), 0).to(tl.float32)
        yi = tl.rsqrt(tl.sum(y*y, 1)/C+EPS)
        yu = y*yi[:, None]
        pa = tl.load(PA+poff, c[:, None] < C, 0)
        pb = tl.load(PB+poff, c[:, None] < C, 0)
        pg = 2.*tl.sigmoid(tl.fma(yu[:, :, None], pa[None, :, :], pb[None, :, :]))
        gp = tl.sum(go*y[:, :, None]*pg, 1)
        dyf = go*p[:, None, :]
        dz = dyf*y[:, :, None]*pg*(1.-.5*pg)
        dyu = tl.sum(dz*pa[None, :, :], 2)
        gy = tl.sum(dyf*pg, 2)+yi[:, None]*(dyu-yu*(tl.sum(dyu*yu, 1)/C)[:, None])
        tl.store(GY+t[:, None]*C+c[None, :], gy,
                 (t[:, None] < M) & (c[None, :] < C))
        tl.store(PART+2*splits*C*4+pid*C*4+poff,
                 tl.sum(dz*yu[:, :, None], 0), c[:, None] < C)
        tl.store(PART+3*splits*C*4+pid*C*4+poff, tl.sum(dz, 0), c[:, None] < C)
    else:
        go = tl.load(GO+t[:, None]*C+c[None, :],
                     (t[:, None] < M) & (c[None, :] < C), 0).to(tl.float32)
        df = go[:, :, None]*p[:, None, :]
        gp = tl.sum(go[:, :, None]*xg, 1)
    gx, ga, gb = _gate_vjp(x, df, a, g, u, inv, C)
    tl.store(GX+off, gx, mask)
    tl.store(GP+t[:, None]*4+s[None, :], gp, t[:, None] < M)
    tl.store(PART+pid*C*4+poff, ga, c[:, None] < C)
    tl.store(PART+splits*C*4+pid*C*4+poff, gb, c[:, None] < C)
