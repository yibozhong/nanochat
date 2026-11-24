"""Fused per-token Chord controls, adapted from lululu39/NextTransformer at 753b013.

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

from nanochat.chord_token_controls import GATE_EPS


@triton.jit
def _project(h, v, W, lane, D: tl.constexpr, k: tl.constexpr):
    wh = tl.load(W + (2 * k) * D + lane, lane < D, 0.0)
    wv = tl.load(W + (2 * k + 1) * D + lane, lane < D, 0.0)
    return tl.sum(h * wh[None, :] + v * wv[None, :], 1)


@triton.jit
def _positive(x):
    # torch softplus with the default linear threshold.
    return tl.where(x > 20., x, tl.maximum(x, 0.) + libdevice.log1p(libdevice.exp(-tl.abs(x))))


@triton.jit
def _parameter_partials(
    H,
    V,
    DZ,
    PW,
    PB,
    ROWS,
    D: tl.constexpr,
    K: tl.constexpr,
    CHUNK: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    tile = tl.program_id(0)
    stream = tl.program_id(1)
    split = tl.program_id(2)
    DZ += stream * ROWS * K
    col = tile * BC + tl.arange(0, BC)
    controls = tl.arange(0, 16)
    sums = tl.full((16, BC), 0, tl.float32)
    biases = tl.full((16,), 0, tl.float32)
    for offset in tl.static_range(0, CHUNK, BR):
        row = split * CHUNK + offset + tl.arange(0, BR)
        ptrs = row[:, None].to(tl.int64) * D + col[None, :]
        valid = (row[:, None] < ROWS) & (col[None, :] < D)
        if stream == 0:
            x = tl.load(H + ptrs, valid, 0)
        else:
            x = tl.load(V + ptrs, valid, 0)
        z = tl.load(
            DZ + row[:, None].to(tl.int64) * K + controls[None, :],
            (row[:, None] < ROWS) & (controls[None, :] < K),
            0.0,
        )
        # Two BF16 limbs retain small FP32 control gradients in tensor-core dot products.
        hi = z.to(tl.bfloat16)
        lo = (z - hi.to(tl.float32)).to(tl.bfloat16)
        sums = tl.dot(tl.trans(hi), x, sums)
        sums = tl.dot(tl.trans(lo), x, sums)
        if tile == 0 and stream == 0:
            biases += tl.sum(z, 0)
    ptrs = ((split * K + controls[:, None].to(tl.int64)) * 2 + stream) * D + col[
        None, :
    ]
    tl.store(PW + ptrs, sums, (controls[:, None] < K) & (col[None, :] < D))
    if tile == 0 and stream == 0:
        tl.store(PB + split * K + controls, biases, controls < K)


@triton.jit
def _finish_parameters(
    PW,
    PB,
    DW,
    DM,
    DS,
    DB,
    DG,
    SPLITS,
    D: tl.constexpr,
    K: tl.constexpr,
    BS: tl.constexpr,
    BC: tl.constexpr,
):
    tile = tl.program_id(0)
    stream = tl.program_id(1)
    k = tl.program_id(2)
    col = tile * BC + tl.arange(0, BC)
    split = tl.arange(0, BS)
    ptrs = ((split[:, None].to(tl.int64) * K + k) * 2 + stream) * D + col[None, :]
    part = tl.load(PW + ptrs, (split[:, None] < SPLITS) & (col[None, :] < D), 0.0)
    tl.store(DW + (2 * k + stream) * D + col, tl.sum(part, 0), col < D)
    if tile == 0 and stream == 0:
        val = tl.sum(tl.load(PB + split * K + k, split < SPLITS, 0.0), 0)
        if k == 0:
            tl.store(DM, val)
        elif k <= 2:
            tl.store(DS + k - 1, val)
        elif K == 6:
            if k == 3:
                tl.store(DB, val)
            else:
                tl.store(DG + k - 4, val)


@triton.jit
def _token_read_forward(
    H,
    V,
    W,
    MU,
    STEPS,
    BETA,
    GAINS,
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
    gate_inv = tl.rsqrt(tl.sum(v * v, 1) / D + EPS)
    gate_v = v * gate_inv[:, None]
    mu = tl.load(MU) + _project(h, gate_v, W, lane, D, 0)
    ap = _positive(tl.load(STEPS) + _project(h, gate_v, W, lane, D, 1))
    al = _positive(tl.load(STEPS + 1) + _project(h, gate_v, W, lane, D, 2))
    if K == 6:
        beta = tl.sigmoid(tl.load(BETA) + _project(h, gate_v, W, lane, D, 3))
        gp = tl.load(GAINS) + _project(h, gate_v, W, lane, D, 4)
        gt = tl.load(GAINS + 1) + _project(h, gate_v, W, lane, D, 5)
    else:
        beta, gp, gt = 0.5, 0.5, 0.5
    dest = C + row.to(tl.int64) * 6
    tl.store(dest, mu, row < ROWS)
    tl.store(dest + 1, ap, row < ROWS)
    tl.store(dest + 2, al, row < ROWS)
    tl.store(dest + 3, beta, row < ROWS)
    tl.store(dest + 4, gp, row < ROWS)
    tl.store(dest + 5, gt, row < ROWS)
    off = mu[:, None] * v
    inv = tl.rsqrt(1.0 + tl.sum(off * off, 1) / D)
    q = ((h + off) * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    scale = tl.rsqrt(tl.sum(q * q, 1) / D + 1.1920928955078125e-7)
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
    gate_inv = tl.rsqrt(tl.sum(v * v, 1) / D + EPS)
    gate_v = v * gate_inv[:, None]
    offc = row.to(tl.int64) * 6
    mu = tl.load(C + offc, row < ROWS, 0.0)
    off = mu[:, None] * v
    inv = tl.rsqrt(1.0 + tl.sum(off * off, 1) / D)
    raw = h + off
    q = (raw * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    scale = tl.rsqrt(tl.sum(q * q, 1) / D + 1.1920928955078125e-7)
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
    grad_gate_v = tl.full((BT, BD), 0.0, tl.float32)
    query_mu_grad = tl.sum(v * d_off, 1)
    for k in tl.static_range(K):
        g = tl.load(DC + offc + k, row < ROWS, 0.0)
        if k == 0:
            g += query_mu_grad
        elif k == 1 or k == 2:
            alpha = tl.load(C + offc + k, row < ROWS, 0.0)
            g *= -libdevice.expm1(-alpha)
        elif k == 3:
            beta = tl.load(C + offc + 3, row < ROWS, 0.5)
            g *= beta * (1.0 - beta)
        dst = DZ + row.to(tl.int64) * K + k
        tl.store(dst, g, row < ROWS)
        # grad_Wv = (dz * gate_inv).T @ v, reusing the original BF16 velocity.
        tl.store(dst + ROWS * K, g * gate_inv, row < ROWS)
        wh = tl.load(W + 2 * k * D + lane, lane < D, 0.0)
        wv = tl.load(W + (2 * k + 1) * D + lane, lane < D, 0.0)
        dh += g[:, None] * wh[None, :]
        grad_gate_v += g[:, None] * wv[None, :]
    radial = tl.sum(grad_gate_v * gate_v, 1) / D
    dv += gate_inv[:, None] * (grad_gate_v - radial[:, None] * gate_v)
    tl.store(DH + ix, dh, mask)
    tl.store(DV + ix, dv, mask)


class TokenRead(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, h, v, weight, mu, steps, beta, gains):
        h, v = h.contiguous(), v.contiguous()
        dim = h.shape[-1]
        count = weight.shape[0]
        rows = h.numel() // dim
        query = torch.empty_like(h)
        controls = weight.new_empty((*h.shape[:-1], 6))
        _token_read_forward[(triton.cdiv(rows, 4),)](
            h,
            v,
            weight,
            mu,
            steps,
            beta if beta is not None else mu,
            gains if gains is not None else mu,
            query,
            controls,
            rows,
            dim,
            count,
            4,
            triton.next_power_of_2(dim),
            GATE_EPS,
            enable_fp_fusion=False,
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
        dz = weight.new_empty((2, rows, count))
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
            enable_fp_fusion=False,
        )
        dw = torch.empty_like(weight)
        dm = weight.new_empty(1)
        ds = weight.new_empty(2)
        db = weight.new_empty(1) if count == 6 else None
        dg = weight.new_empty(2) if count == 6 else None
        chunk = 256
        cols = 64
        splits = triton.cdiv(rows, chunk)
        pw = weight.new_empty((splits, count, 2, dim))
        pb = weight.new_empty((splits, count))
        _parameter_partials[(triton.cdiv(dim, cols), 2, splits)](
            h, v, dz, pw, pb, rows, dim, count, chunk, 32, cols
        )
        _finish_parameters[(triton.cdiv(dim, cols), 2, count)](
            pw,
            pb,
            dw,
            dm,
            ds,
            db if db is not None else dm,
            dg if dg is not None else dm,
            splits,
            dim,
            count,
            triton.next_power_of_2(splits),
            cols,
        )
        return dh, dv, dw, dm, ds, db, dg
