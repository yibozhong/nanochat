"""Three-control Chord kernels adapted from NextTransformer (98321666).

MIT License

Copyright (c) 2026 NextTransformer contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from nanochat.chord_token_read import _project, _positive, _parameter_partials
from nanochat.chord_single_step import GATE_EPS


@triton.jit
def _token_read_forward(
    H,
    V,
    W,
    MU,
    STEPS,
    BETA,
    Q,
    C,
    ROWS,
    D: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    lane = tl.arange(0, BD)
    ix = row[:, None].to(tl.int64) * D + lane[None, :]
    mask = (row[:, None] < ROWS) & (lane[None, :] < D)
    h = tl.load(H + ix, mask, 0.0).to(tl.float32)
    v = tl.load(V + ix, mask, 0.0).to(tl.float32)
    gate_inv = tl.rsqrt(tl.sum(h * h, 1) / D + EPS)
    mu = tl.load(MU) + gate_inv * _project(h, v, W, lane, D, 0, 1)
    ap = _positive(tl.load(STEPS) + gate_inv * _project(h, v, W, lane, D, 1, 1))
    if K == 3:
        beta = tl.sigmoid(tl.load(BETA) + gate_inv * _project(h, v, W, lane, D, 2, 1))
    else:
        beta = 0.5
    dest = C + row.to(tl.int64) * 3
    tl.store(dest, mu, row < ROWS)
    tl.store(dest + 1, ap, row < ROWS)
    tl.store(dest + 2, beta, row < ROWS)
    off = mu[:, None] * v
    inv = tl.rsqrt(1.0 + tl.sum(off * off, 1) / D)
    q = ((h + off) * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    scale = 1.0 / tl.sqrt(tl.sum(q * q, 1) / D + 1.1920928955078125e-7)
    tl.store(Q + ix, q * scale[:, None], mask)


@triton.jit
def _token_read_backward(
    H,
    V,
    W,
    C,
    DQ,
    DC,
    DH,
    DV,
    DZ,
    ROWS,
    D: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    lane = tl.arange(0, BD)
    ix = row[:, None].to(tl.int64) * D + lane[None, :]
    mask = (row[:, None] < ROWS) & (lane[None, :] < D)
    h = tl.load(H + ix, mask, 0.0).to(tl.float32)
    v = tl.load(V + ix, mask, 0.0).to(tl.float32)
    gate_inv = tl.rsqrt(tl.sum(h * h, 1) / D + EPS)
    offc = row.to(tl.int64) * 3
    mu = tl.load(C + offc, row < ROWS, 0.0)
    off = mu[:, None] * v
    inv = tl.rsqrt(1.0 + tl.sum(off * off, 1) / D)
    raw = h + off
    q = (raw * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    scale = 1.0 / tl.sqrt(tl.sum(q * q, 1) / D + 1.1920928955078125e-7)
    normed = q * scale[:, None]
    dy = tl.load(DQ + ix, mask, 0.0).to(tl.float32)
    dq = (
        ((dy - normed * (tl.sum(dy * normed, 1) / D)[:, None]) * scale[:, None])
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    direct = dq * inv[:, None]
    coeff = tl.sum(dq * raw, 1) / D * inv * inv * inv
    d_off = direct - off * coeff[:, None]
    dh = direct
    dv = mu[:, None] * d_off
    grad_gate_h = tl.full((BT, BD), 0.0, tl.float32)
    query_mu_grad = tl.sum(v * d_off, 1)
    grad_row = DZ + row.to(tl.int64) * (K + 1)
    tl.store(grad_row + K, gate_inv, row < ROWS)
    for k in tl.static_range(K):
        g = tl.load(DC + offc + k, row < ROWS, 0.0)
        if k == 0:
            g += query_mu_grad
        elif k == 1:
            alpha = tl.load(C + offc + k, row < ROWS, 0.0)
            g *= -libdevice.expm1(-alpha)
        elif k == 2:
            beta = tl.load(C + offc + k, row < ROWS, 0.0)
            g *= beta * (1.0 - beta)
        tl.store(grad_row + k, g, row < ROWS)
        wh = tl.load(W + k * D + lane, lane < D, 0.0)
        grad_gate_h += g[:, None] * wh[None, :]
    gate_h = h * gate_inv[:, None]
    radial = tl.sum(grad_gate_h * gate_h, 1) / D
    dh += gate_inv[:, None] * (grad_gate_h - radial[:, None] * gate_h)
    tl.store(DH + ix, dh, mask)
    tl.store(DV + ix, dv, mask)


@triton.jit
def _finish_parameters(
    PW,
    PB,
    DW,
    DM,
    DS,
    DB,
    SPLITS,
    D: tl.constexpr,
    K: tl.constexpr,
    BS: tl.constexpr,
    BC: tl.constexpr,
):
    tile = tl.program_id(0)
    k = tl.program_id(1)
    col = tile * BC + tl.arange(0, BC)
    split = tl.arange(0, BS)
    ptrs = (split[:, None].to(tl.int64) * K + k) * D + col[None, :]
    part = tl.load(PW + ptrs, (split[:, None] < SPLITS) & (col[None, :] < D), 0.0)
    tl.store(DW + k * D + col, tl.sum(part, 0), col < D)
    if tile == 0:
        val = tl.sum(tl.load(PB + split * K + k, split < SPLITS, 0.0), 0)
        if k == 0:
            tl.store(DM, val)
        elif k == 1:
            tl.store(DS + k - 1, val)
        elif K == 3:
            tl.store(DB, val)


class TokenRead(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, h, v, weight, mu, steps, beta):
        h, v = h.contiguous(), v.contiguous()
        dim = h.shape[-1]
        count = weight.shape[0]
        rows = h.numel() // dim
        query = torch.empty_like(h)
        controls = weight.new_empty((*h.shape[:-1], 3))
        _token_read_forward[(triton.cdiv(rows, 4),)](
            h,
            v,
            weight,
            mu,
            steps,
            beta if beta is not None else mu,
            query,
            controls,
            rows,
            dim,
            count,
            4,
            triton.next_power_of_2(dim),
            GATE_EPS,
        )
        ctx.save_for_backward(h, v, weight, controls)
        return query, controls

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gquery, gcontrols):
        h, v, weight, controls = ctx.saved_tensors
        dim = h.shape[-1]
        count = weight.shape[0]
        rows = h.numel() // dim
        dh, dv = torch.empty_like(h), torch.empty_like(v)
        dz = weight.new_empty((rows, count + 1))
        tile = min(4, max(1, 4096 // triton.next_power_of_2(dim)))
        _token_read_backward[(triton.cdiv(rows, tile),)](
            h,
            v,
            weight,
            controls,
            gquery.contiguous(),
            gcontrols.contiguous(),
            dh,
            dv,
            dz,
            rows,
            dim,
            count,
            tile,
            triton.next_power_of_2(dim),
            GATE_EPS,
        )
        dw = torch.empty_like(weight)
        dm = weight.new_empty(1)
        ds = weight.new_empty(1)
        db = weight.new_empty(1) if count == 3 else None
        chunk = 256
        cols = 64
        splits = triton.cdiv(rows, chunk)
        pw = weight.new_empty((splits, count, dim))
        pb = weight.new_empty((splits, count))
        _parameter_partials[(triton.cdiv(dim, cols), 1, splits)](
            h, h, dz, pw, pb, rows, dim, count, chunk, 32, cols, True, 1
        )
        _finish_parameters[(triton.cdiv(dim, cols), count)](
            pw,
            pb,
            dw,
            dm,
            ds,
            db if db is not None else dm,
            splits,
            dim,
            count,
            triton.next_power_of_2(splits),
            cols,
        )
        return dh, dv, dw, dm, ds, db


@triton.jit
def _token_write(
    H,
    V,
    F,
    P,
    NH,
    NV,
    GH,
    GV,
    DH,
    DV,
    DF,
    DP,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    BACKWARD: tl.constexpr,
    FORCE_FLOOR_SQ: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    idx = row.to(tl.int64) * D + lane
    h = tl.load(H + idx, lane < D, 0).to(tl.float32)
    v = tl.load(V + idx, lane < D, 0).to(tl.float32)
    force = tl.load(F + idx, lane < D, 0).to(tl.float32)
    alpha = tl.load(P + 3 * row + 1)
    beta = tl.load(P + 3 * row + 2)
    f2 = tl.sum(force * force, 0) / D
    fi = tl.rsqrt(tl.maximum(f2, FORCE_FLOOR_SQ))
    target = force * fi
    delta = target - h
    mixed = h + alpha * delta
    m2 = tl.sum(mixed * mixed, 0) / D
    mi = tl.rsqrt(tl.maximum(m2, 1e-12))
    nxt = mixed * mi
    chord = nxt - h
    velocity = beta * v + (1.0 - beta) * chord
    radial = tl.sum(velocity * nxt, 0) / D
    if BACKWARD:
        gh = tl.load(GH + idx, lane < D, 0).to(tl.float32)
        gv = tl.load(GV + idx, lane < D, 0).to(tl.float32)
        dot = tl.sum(gv * nxt, 0) / D
        g_velocity = gv - dot * nxt
        g_chord = (1.0 - beta) * g_velocity
        g_nxt = gh + g_chord - radial * gv - dot * velocity
        g_mixed = mi * (
            g_nxt - tl.where(m2 >= 1e-12, nxt * (tl.sum(g_nxt * nxt, 0) / D), 0.0)
        )
        g_delta = alpha * g_mixed
        g_v = beta * g_velocity
        g_force = fi * (
            g_delta
            - tl.where(f2 >= FORCE_FLOOR_SQ, target * (tl.sum(g_delta * target, 0) / D), 0.0)
        )
        tl.store(DH + idx, g_mixed - g_delta - g_chord, lane < D)
        tl.store(DV + idx, g_v, lane < D)
        tl.store(DF + idx, g_force, lane < D)
        tl.store(DP + 3 * row, 0.0)
        tl.store(DP + 3 * row + 1, tl.sum(g_mixed * delta, 0))
        tl.store(DP + 3 * row + 2, tl.sum(g_velocity * (v - chord), 0))
    else:
        tl.store(NH + idx, nxt, lane < D)
        tl.store(NV + idx, velocity - radial * nxt, lane < D)


class TokenWrite(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, h, v, force, controls, floor):
        ctx.force_floor_sq = floor ** 2
        h, v, force = h.contiguous(), v.contiguous(), force.contiguous()
        width = h.size(-1)
        rows = h.numel() // width
        nh, nv = torch.empty_like(h), torch.empty_like(v)
        _token_write[(rows,)](
            h,
            v,
            force,
            controls,
            nh,
            nv,
            None,
            None,
            None,
            None,
            None,
            None,
            width,
            triton.next_power_of_2(width),
            False, ctx.force_floor_sq,
        )
        ctx.save_for_backward(h, v, force, controls)
        return nh, nv

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gh, gv):
        h, v, force, controls = ctx.saved_tensors
        width = h.size(-1)
        rows = h.numel() // width
        dh, dv, df = torch.empty_like(h), torch.empty_like(v), torch.empty_like(force)
        dc = torch.empty_like(controls)
        _token_write[(rows,)](
            h,
            v,
            force,
            controls,
            None,
            None,
            gh.contiguous(),
            gv.contiguous(),
            dh,
            dv,
            df,
            dc,
            width,
            triton.next_power_of_2(width),
            True, ctx.force_floor_sq,
        )
        return dh, dv, df, dc, None
