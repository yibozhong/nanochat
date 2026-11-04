# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Local specialization/adaptation; provenance and MIT license in
# research/connections/triton-moments/upstream. Optional FLA features are
# retained in the vendored kernels but the local wrapper accepts causal MHA only.
import torch
import triton
import triton.language as tl

@triton.jit
def exp2(x): return tl.math.exp2(x.to(tl.float32))
@triton.jit
def log2(x): return tl.log2(x.to(tl.float32))
@triton.jit
def unflatten_program_id(X: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    return pid % X, pid // X

@triton.jit
def phi(x, MODE: tl.constexpr):
    x = x.to(tl.float32)
    if MODE == 3:
        return x * tl.sigmoid(x)
    if MODE == 4:
        return tl.abs(x)
    return x * x

@triton.jit
def phi_grad(x, MODE: tl.constexpr):
    x = x.to(tl.float32)
    if MODE == 3:
        s = tl.sigmoid(x)
        return s * (1. + x * (1. - s))
    if MODE == 4:
        # Match torch.abs: the chosen subgradient at zero is zero.
        return tl.where(x > 0., 1., tl.where(x < 0., -1., 0.))
    return 2. * x

@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'USE_SINK_BIAS': lambda args: args['sink_bias'] is not None,
    'USE_WINDOW': lambda args: args['W'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def support_fwd_kernel(
    q,
    k,
    v,
    o,
    stat,
    log_support,
    raw,
    MODE: tl.constexpr,
    g_cumsum,
    sink_bias,
    lse,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_SINK_BIAS: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t = unflatten_program_id(tl.cdiv(V, BV))
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_q = o_q < T
    p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    p_o = o + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    p_lse = lse + bos * HQ + i_hq + o_q * HQ

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
    # [BT, BV]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    b_z = tl.zeros([BT, BV], dtype=tl.float32)

    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)
    b_sq = tl.zeros([BT], dtype=tl.float32)

    if USE_G:
        p_g = g_cumsum + bos * HQ + i_hq + o_q * HQ
        b_gq = tl.load(p_g, mask=m_q, other=0.0).to(tl.float32)
    else:
        b_gq = None

    if USE_SINK_BIAS:
        b_sink_bias = tl.load(sink_bias + i_hq).to(tl.float32)
    else:
        b_sink_bias = None

    # for sliding window, skip key blocks that are entirely outside the window.
    # the earliest key position any query in this block needs: max(0, i_t*BT - W + 1)
    i_start = tl.maximum((i_t * BT - W + 1) // BS * BS, 0) if USE_WINDOW else 0

    for i_s in range(i_start, i_t * BT, BS):
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]
        # [BK, BS]
        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        # [BS, BV]
        b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2

        if USE_G:
            b_gk = tl.load(g_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]

        if USE_WINDOW:
            b_s = tl.where((o_q[:, None] - o_k[None, :] < W) & m_k[None, :], b_s, float('-inf'))

        # [BT, BS]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        # keep the online softmax pivot finite for rows that still have no valid key.
        # this matches sglang's masked-row stabilization and avoids -inf - (-inf) = NaN.
        b_mw = tl.where(b_m == float('-inf'), 0., b_m)
        b_r = exp2(b_mp - b_mw)
        b_p = exp2(b_s - b_mw[:, None])
        # [BT]
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_sq = b_sq * b_r * b_r + tl.sum(b_p * b_p, 1)
        # [BT, BV]
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

        b_z = b_z * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), phi(b_v, MODE).to(b_v.dtype))
        b_mp = b_m

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        # [BS]
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]

        # [BK, BS]
        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        # [BS, BV]
        b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2

        if USE_G:
            b_gk = tl.load(g_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]

        m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        if USE_WINDOW:
            m_s = m_s & (o_q[:, None] - o_k[None, :] < W)
        b_s = tl.where(m_s, b_s, float('-inf'))

        # [BT]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_mw = tl.where(b_m == float('-inf'), 0., b_m)
        b_r = exp2(b_mp - b_mw)
        b_p = exp2(b_s - b_mw[:, None])
        # [BT]
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_sq = b_sq * b_r * b_r + tl.sum(b_p * b_p, 1)
        # [BT, BV]
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
        b_z = b_z * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), phi(b_v, MODE).to(b_v.dtype))
        b_mp = b_m

    if USE_SINK_BIAS:
        # when a row has no valid key at all, b_m is still -inf here.
        # use a finite pivot before merging the sink-bias mass so lse becomes
        # the sink-bias logit instead of hitting the -inf + inf = NaN path.
        b_m = tl.where(b_m == float('-inf'), 0., b_m)
        # denominator-only sink-bias update (matches GPT-OSS / sglang):
        # the bias logit augments the softmax normalizer without contributing
        # to the value matmul.
        b_acc += exp2(b_sink_bias - b_m)

    b_o = b_o / b_acc[:, None]
    b_z = b_z / b_acc[:, None]
    # Center with the same BF16 phi evaluation as the V side. Keep the
    # accumulators on chip until centering; no BF16 raw-moment subtraction.
    b_mean = b_o.to(o.dtype.element_ty)
    if MODE == 1:
        b_stat = b_z
    else:
        b_stat = b_z - phi(b_mean, MODE).to(b_mean.dtype).to(tl.float32)
        if MODE == 2:
            b_stat = tl.maximum(b_stat, 0.)
    p_stat = stat + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    p_raw = raw + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    tl.store(p_stat, b_stat, mask=m_q[:, None] & (o_v[None, :] < V))
    tl.store(p_raw, b_z, mask=m_q[:, None] & (o_v[None, :] < V))
    tl.store(log_support + bos * HQ + i_hq + o_q * HQ, 2. * tl.log(b_acc) - tl.log(b_sq), mask=m_q)
    b_m += log2(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_q[:, None] & (o_v[None, :] < V))
    if i_v == 0:
        tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), mask=m_q)


@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'USE_WINDOW': lambda args: args['W'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def support_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dz,
    log_support,
    dlog_support,
    MODE: tl.constexpr,
    dq,
    dg_cumsum,
    g_cumsum,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t = unflatten_program_id(tl.cdiv(V, BV))
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
    # NOTE: we must multiply RCP_LN2 after tl.dot for high precision
    RCP_LN2: tl.constexpr = 1.4426950216

    o_q = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_q = o_q < T
    p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    p_dq = dq + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    p_do = do + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    p_lse = lse + bos * HQ + i_hq + o_q * HQ
    p_delta = delta + bos * HQ + i_hq + o_q * HQ

    # [BT, BK]
    b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
    # [BT, BV]
    b_do = tl.load(p_do, mask=m_q[:, None] & (o_v[None, :] < V), other=0.0).to(v.dtype.element_ty)
    b_dz = tl.load(dz + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :], mask=m_q[:, None] & (o_v[None, :] < V), other=0.).to(v.dtype.element_ty)
    # [BT]
    b_lse = tl.load(p_lse, mask=m_q, other=0.0)
    b_delta = tl.load(p_delta, mask=m_q, other=0.0)
    b_neff = tl.exp(tl.load(log_support + bos * HQ + i_hq + o_q * HQ, mask=m_q, other=0.))
    b_dlog = tl.load(dlog_support + bos * HQ + i_hq + o_q * HQ, mask=m_q, other=0.)
    # A singleton has constant support=1, hence an exactly zero adjoint.
    b_dlog = tl.where(o_q == 0, 0., b_dlog)

    # [BT, BK]
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    if USE_G:
        b_dg = tl.zeros([BT], dtype=tl.float32)
        p_gq = g_cumsum + bos * HQ + i_hq + o_q * HQ
        b_gq = tl.load(p_gq, mask=m_q, other=0.0).to(tl.float32)
    else:
        b_gq = None
        b_dg = None

    i_start = tl.maximum((i_t * BT - W + 1) // BS * BS, 0) if USE_WINDOW else 0

    for i_s in range(i_start, i_t * BT, BS):
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_v[:, None] + o_k[None, :] * (H*V)

        # [BK, BS]
        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        # [BV, BS]
        b_v = tl.load(p_v, mask=(o_v[:, None] < V) & m_k[None, :], other=0.0)
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        if USE_G:
            b_gk = tl.load(g_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]

        if USE_WINDOW:
            b_s = tl.where((o_q[:, None] - o_k[None, :] < W) & m_k[None, :], b_s, float('-inf'))
        b_p = exp2(b_s - b_lse[:, None])
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_do, b_v) + tl.dot(b_dz, phi(b_v, MODE).to(b_v.dtype))
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        b_ds += 2. * b_dlog[:, None] * b_p * (1. - b_neff[:, None] * b_p)
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dq = tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k), b_dq)
        if USE_G:
            b_dg += tl.sum(b_ds, 1)

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        # [BS]
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_v[:, None] + o_k[None, :] * (H*V)

        # [BK, BS]
        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        # [BV, BS]
        b_v = tl.load(p_v, mask=(o_v[:, None] < V) & m_k[None, :], other=0.0)
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2

        if USE_G:
            p_gk = g_cumsum + bos * HQ + i_hq + o_k * HQ
            b_gk = tl.load(p_gk, mask=m_k, other=0.0).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]
        if USE_WINDOW:
            b_p = tl.where(
                (o_q[:, None] >= o_k[None, :]) & (o_q[:, None] - o_k[None, :] < W) & m_k[None, :],
                exp2(b_s - b_lse[:, None]), 0
            )
        else:
            b_p = tl.where((o_q[:, None] >= o_k[None, :]) & m_k[None, :], exp2(b_s - b_lse[:, None]), 0)

        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_do, b_v) + tl.dot(b_dz, phi(b_v, MODE).to(b_v.dtype))
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        b_ds += 2. * b_dlog[:, None] * b_p * (1. - b_neff[:, None] * b_p)
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dq = tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k), b_dq)
        if USE_G:
            b_dg += tl.sum(b_ds, 1)

    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_q[:, None] & (o_d[None, :] < K))
    if USE_G:
        p_dg = dg_cumsum + bos * HQ + i_hq + o_q * HQ
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_q)


@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'USE_WINDOW': lambda args: args['W'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def support_bwd_kernel_dkv(
    q,
    k,
    v,
    g_cumsum,
    lse,
    delta,
    do,
    dz,
    log_support,
    dlog_support,
    MODE: tl.constexpr,
    dk,
    dv,
    dg_cumsum,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t = unflatten_program_id(tl.cdiv(V, BV))
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    o_k = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < T
    p_k = k + (bos * H + i_h) * K + o_k[:, None] * (H*K) + o_d[None, :]
    p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]
    p_dk = dk + (bos * HQ + i_hq) * K + o_k[:, None] * (HQ*K) + o_d[None, :]
    p_dv = dv + (bos * HQ + i_hq) * V + o_k[:, None] * (HQ*V) + o_v[None, :]

    # [BT, BK]
    b_k = tl.load(p_k, mask=m_k[:, None] & (o_d[None, :] < K), other=0.0)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    # [BT, BV]
    b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    b_dzv = tl.zeros([BT, BV], dtype=tl.float32)

    if USE_G:
        p_gk = g_cumsum + bos * HQ + i_hq + o_k * HQ
        b_gk = tl.load(p_gk, mask=m_k, other=0.0).to(tl.float32)
        b_dg = tl.zeros([BT], dtype=tl.float32)
    else:
        b_gk = None
        b_dg = None

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        # [BS]
        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
        p_do = do + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
        p_lse = lse + bos * HQ + i_hq + o_q * HQ
        p_delta = delta + bos * HQ + i_hq + o_q * HQ

        # [BS, BK]
        b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
        # [BS, BV]
        b_do = tl.load(p_do, mask=m_q[:, None] & (o_v[None, :] < V), other=0.0).to(v.dtype.element_ty)
        b_dz = tl.load(dz + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :], mask=m_q[:, None] & (o_v[None, :] < V), other=0.).to(v.dtype.element_ty)
        # [BS]
        b_lse = tl.load(p_lse, mask=m_q, other=0.0)
        b_delta = tl.load(p_delta, mask=m_q, other=0.0)
        b_neff = tl.exp(tl.load(log_support + bos * HQ + i_hq + o_q * HQ, mask=m_q, other=0.))
        b_dlog = tl.load(dlog_support + bos * HQ + i_hq + o_q * HQ, mask=m_q, other=0.)
        # A singleton has constant support=1, hence an exactly zero adjoint.
        b_dlog = tl.where(o_q == 0, 0., b_dlog)
        # [BT, BS]
        b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2
        if USE_G:
            p_gq = g_cumsum + bos * HQ + i_hq + o_q * HQ
            b_gq = tl.load(p_gq, mask=m_q, other=0.0).to(tl.float32)
            b_s += b_gq[None, :] - b_gk[:, None]
        if USE_WINDOW:
            b_p = tl.where(
                (o_k[:, None] <= o_q[None, :]) & (o_q[None, :] - o_k[:, None] < W) & m_q[None, :],
                exp2(b_s - b_lse[None, :]), 0
            )
        else:
            b_p = tl.where((o_k[:, None] <= o_q[None, :]) & m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        # [BT, BS] @ [BS, BV] -> [BT, BV]
        b_dv = tl.dot(b_p.to(b_do.dtype), b_do, b_dv)
        b_dzv = tl.dot(b_p.to(b_do.dtype), b_dz, b_dzv)
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_v, tl.trans(b_do)) + tl.dot(phi(b_v, MODE).to(b_v.dtype), tl.trans(b_dz))
        # [BT, BS]
        b_ds = b_p * (b_dp - b_delta[None, :])
        b_ds += 2. * b_dlog[None, :] * b_p * (1. - b_neff[None, :] * b_p)
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dk = tl.dot(b_ds.to(b_q.dtype), b_q, b_dk)
        if USE_G:
            b_dg -= tl.sum(b_ds, 1)

    # for sliding window, limit the range of future query blocks to process.
    # a key at position k_pos can only be attended to by queries at positions [k_pos, k_pos + W - 1].
    # so we only need queries up to (i_t + 1) * BT - 1 + W - 1.
    i_end = min(tl.cdiv(T, BS) * BS, (i_t + 1) * BT + W - 1) if USE_WINDOW else tl.cdiv(T, BS) * BS

    for i_s in range((i_t + 1) * BT, i_end, BS):
        # [BS]
        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
        p_do = do + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
        p_lse = lse + bos * HQ + i_hq + o_q * HQ
        p_delta = delta + bos * HQ + i_hq + o_q * HQ

        # [BS, BK]
        b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
        # [BS, BV]
        b_do = tl.load(p_do, mask=m_q[:, None] & (o_v[None, :] < V), other=0.0).to(v.dtype.element_ty)
        b_dz = tl.load(dz + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :], mask=m_q[:, None] & (o_v[None, :] < V), other=0.).to(v.dtype.element_ty)
        # [BS]
        b_lse = tl.load(p_lse, mask=m_q, other=0.0)
        b_delta = tl.load(p_delta, mask=m_q, other=0.0)
        b_neff = tl.exp(tl.load(log_support + bos * HQ + i_hq + o_q * HQ, mask=m_q, other=0.))
        b_dlog = tl.load(dlog_support + bos * HQ + i_hq + o_q * HQ, mask=m_q, other=0.)
        # A singleton has constant support=1, hence an exactly zero adjoint.
        b_dlog = tl.where(o_q == 0, 0., b_dlog)
        # [BT, BS]
        b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2
        if USE_G:
            p_gq = g_cumsum + bos * HQ + i_hq + o_q * HQ
            b_gq = tl.load(p_gq, mask=m_q, other=0.0).to(tl.float32)
            b_s += b_gq[None, :] - b_gk[:, None]
        if USE_WINDOW:
            b_p = tl.where((o_q[None, :] - o_k[:, None] < W) & m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        else:
            b_p = tl.where(m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        # [BT, BS] @ [BS, BV] -> [BT, BV]
        b_dv = tl.dot(b_p.to(b_do.dtype), b_do, b_dv)
        b_dzv = tl.dot(b_p.to(b_do.dtype), b_dz, b_dzv)
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_v, tl.trans(b_do)) + tl.dot(phi(b_v, MODE).to(b_v.dtype), tl.trans(b_dz))
        # [BT, BS]
        b_ds = b_p * (b_dp - b_delta[None, :])
        b_ds += 2. * b_dlog[None, :] * b_p * (1. - b_neff[None, :] * b_p)
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dk = tl.dot(b_ds.to(b_q.dtype), b_q, b_dk)
        if USE_G:
            b_dg -= tl.sum(b_ds, 1)

    b_dv += b_dzv * phi_grad(b_v, MODE)
    b_dk = b_dk * scale
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_k[:, None] & (o_d[None, :] < K))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_k[:, None] & (o_v[None, :] < V))
    if USE_G:
        p_dg = dg_cumsum + bos * HQ + i_hq + o_k * HQ
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_k)


