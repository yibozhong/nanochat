# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under Apache-2.0; see research/connections/stream-scaling/NVIDIA-LICENSE.
# Adapted from hash-pinned TE2.20.2 common/triton/mhc.py for benchmarking only.
# Changes: remove the n==4 compiler assumption and autotuner; choose bounded
# token tiles in the wrapper. Arithmetic/20 log-space iterations unchanged.
import triton
import triton.language as tl

@triton.jit
def _mhc_sinkhorn_fwd_fused_recompute(
    x_ptr,  # (M, n*n)
    output_ptr,  # (M, n*n)
    stride_xm,
    stride_xn,
    stride_out_m,
    stride_out_n,
    M,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    iters,
):
    pid = tl.program_id(0)

    tl.static_assert(BLOCK_SIZE % (n * n) == 0, "BLOCK_SIZE must be divisible by n*n")
    tl.assume(M > 0 and iters > 0)

    BATCH_SIZE: tl.constexpr = BLOCK_SIZE // (n * n)

    offs_batch = pid * BATCH_SIZE + tl.arange(0, BATCH_SIZE)
    offs_nn = tl.arange(0, n * n)
    mask_batch = offs_batch < M

    x_ptrs = x_ptr + offs_batch[:, None] * stride_xm + offs_nn[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask_batch[:, None], other=0.0)  # (BATCH_SIZE, n*n)
    x = tl.reshape(x, (BATCH_SIZE, n, n))  # (BATCH_SIZE, n, n)

    log_mu = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)
    log_nu = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)

    f = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)
    g = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)

    for _ in range(iters):
        # Update f: logsumexp over the column dimension (1)
        f = x + g[:, None, :]  # Broadcast g to (BATCH_SIZE, n, n)
        f_max = tl.max(f, axis=2)
        f = tl.log(tl.sum(tl.exp(f - f_max[:, :, None]), axis=2))  # logsumexp over columns
        f = log_mu - f - f_max

        # Update g: logsumexp over the row dimension (2)
        g = x + f[:, :, None]  # Broadcast f to (BATCH_SIZE, n, n)
        g_max = tl.max(g, axis=1)
        g = tl.log(tl.sum(tl.exp(g - g_max[:, None, :]), axis=1))  # logsumexp over rows
        g = log_nu - g - g_max

    log_P = f[:, :, None] + x + g[:, None, :]
    log_P = tl.reshape(
        log_P,
        (
            BATCH_SIZE,
            n * n,
        ),
    )
    P = tl.exp(log_P)

    output_ptrs = output_ptr + offs_batch[:, None] * stride_out_m + offs_nn[None, :] * stride_out_n
    tl.store(output_ptrs, P, mask=mask_batch[:, None])


@triton.jit
def _mhc_sinkhorn_bwd_fused_recompute(
    grad_out_ptr,
    output_ptr,
    grad_x_ptr,
    x_ptr,
    hist_f_ptr,
    hist_g_ptr,
    stride_grad_out_m,
    stride_grad_out_n,
    stride_out_m,
    stride_out_n,
    stride_grad_xm,
    stride_grad_xn,
    stride_xm,
    stride_xn,
    M,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    iters,
):
    pid = tl.program_id(0)

    tl.static_assert(BLOCK_SIZE % (n * n) == 0, "BLOCK_SIZE must be divisible by n*n")
    tl.assume(M > 0 and iters > 0)

    BATCH_SIZE: tl.constexpr = BLOCK_SIZE // (n * n)  # Assume there's no remainder for simplicity

    offs_batch = pid * BATCH_SIZE + tl.arange(0, BATCH_SIZE)
    offs_nn = tl.arange(0, n * n)
    offs_n_hist = tl.arange(0, n)
    mask_batch = offs_batch < M

    x_ptrs = x_ptr + offs_batch[:, None] * stride_xm + offs_nn[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask_batch[:, None], other=0.0)  # (BATCH_SIZE, n*n)
    x = tl.reshape(x, (BATCH_SIZE, n, n))  # (BATCH_SIZE, n, n)

    P_ptrs = output_ptr + offs_batch[:, None] * stride_out_m + offs_nn[None, :] * stride_out_n
    P = tl.load(P_ptrs, mask=mask_batch[:, None], other=0.0)  # (BATCH_SIZE, n*n)
    P = tl.reshape(P, (BATCH_SIZE, n, n))

    grad_out_ptrs = (
        grad_out_ptr
        + offs_batch[:, None] * stride_grad_out_m
        + offs_nn[None, :] * stride_grad_out_n
    )
    grad_out = tl.load(grad_out_ptrs, mask=mask_batch[:, None], other=0.0)  # (BATCH_SIZE, n*n)
    grad_out = tl.reshape(grad_out, (BATCH_SIZE, n, n))  # (BATCH_SIZE, n, n)

    sbn = M * n

    # Recompute the full history of f and g
    log_mu = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)
    log_nu = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)

    f = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)
    g = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)

    f_hist_ptrs = hist_f_ptr + offs_batch[:, None] * n + offs_n_hist[None, :]
    g_hist_ptrs = hist_g_ptr + offs_batch[:, None] * n + offs_n_hist[None, :]
    tl.store(f_hist_ptrs, f, mask=mask_batch[:, None])
    tl.store(g_hist_ptrs, g, mask=mask_batch[:, None])

    for iter_idx in range(iters):
        # Update f: logsumexp over the column dimension (1)
        f = x + g[:, None, :]  # Broadcast g to (BATCH_SIZE, n, n)
        f_max = tl.max(f, axis=2)
        f = tl.log(tl.sum(tl.exp(f - f_max[:, :, None]), axis=2))  # logsumexp over columns
        f = log_mu - f - f_max

        f_hist_ptrs = (
            hist_f_ptr + (iter_idx + 1) * sbn + offs_batch[:, None] * n + offs_n_hist[None, :]
        )
        tl.store(f_hist_ptrs, f, mask=mask_batch[:, None])

        # Update g: logsumexp over the row dimension (2)
        g = x + f[:, :, None]  # Broadcast f to (BATCH_SIZE, n, n)
        g_max = tl.max(g, axis=1)
        g = tl.log(tl.sum(tl.exp(g - g_max[:, None, :]), axis=1))  # logsumexp over rows
        g = log_nu - g - g_max

        g_hist_ptrs = (
            hist_g_ptr + (iter_idx + 1) * sbn + offs_batch[:, None] * n + offs_n_hist[None, :]
        )
        tl.store(g_hist_ptrs, g, mask=mask_batch[:, None])

    # Backward pass
    grad_log_P = grad_out * P  # (BATCH_SIZE, n, n)
    zeros = tl.zeros_like(grad_log_P)
    grad_g = tl.sum(grad_log_P, axis=1)  # (BATCH_SIZE, n)
    grad_x = grad_log_P

    g_hist_ptrs = hist_g_ptr + iters * sbn + offs_batch[:, None] * n + offs_n_hist[None, :]
    g = tl.load(g_hist_ptrs, mask=mask_batch[:, None], other=0.0)
    g = tl.reshape(g, (BATCH_SIZE, n))

    for iter_idx in range(iters, 0, -1):
        f_hist_ptrs = hist_f_ptr + iter_idx * sbn + offs_batch[:, None] * n + offs_n_hist[None, :]
        f = tl.load(f_hist_ptrs, mask=mask_batch[:, None], other=0.0)
        f = tl.reshape(f, (BATCH_SIZE, n))

        g_hist_ptrs = (
            hist_g_ptr + (iter_idx - 1) * sbn + offs_batch[:, None] * n + offs_n_hist[None, :]
        )
        g_next = tl.load(g_hist_ptrs, mask=mask_batch[:, None], other=0.0)
        g_next = tl.reshape(g_next, (BATCH_SIZE, n))

        term_g = -grad_g[:, None, :] * tl.exp(f[:, :, None] + x + g[:, None, :])
        grad_f = tl.sum(term_g + grad_log_P, axis=2)  # (BATCH_SIZE, n)
        # Only the last iteration's f will contribute to gradients with both grad_g1 and grad_log_P
        grad_log_P = zeros  # Zero out grad_log_P for next iterations

        g = g_next

        term_f = -grad_f[:, :, None] * tl.exp(f[:, :, None] + x + g[:, None, :])
        grad_g = tl.sum(term_f, axis=1)  # (BATCH_SIZE, n)

        grad_x += term_f + term_g

    grad_x_ptrs = (
        grad_x_ptr + offs_batch[:, None] * stride_grad_xm + offs_nn[None, :] * stride_grad_xn
    )
    tl.store(
        grad_x_ptrs,
        tl.reshape(
            grad_x,
            (
                BATCH_SIZE,
                n * n,
            ),
        ),
        mask=mask_batch[:, None],
    )
