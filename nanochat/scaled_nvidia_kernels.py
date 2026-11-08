# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See research/connections/scaled-hc-nvidia/LICENSE.NVIDIA.
# Modified 2026-11-07: static channel/input-stream scale and its gradient,
# FP32 scaled-input routing gradient, branch-last forward, standalone autotune.
"""Adapted directly from Transformer Engine2.20.2 expand/combine kernels.

Keep NVIDIA's split/FMA layout and one-pass backward over all channels.
No scaled-state bank, atomic parameter update, or extra activation scan.
Source hashes and a complete patch are in research/connections/scaled-hc-nvidia.
"""
import itertools
import triton
import triton.language as tl


def _forward_configs():
    return [triton.Config(dict(BLOCK_SIZE_M=m,BLOCK_SIZE_C=256),num_warps=w,num_stages=s)
            for m,w,s in itertools.product((2,4),(1,2),(1,2,3))]


def _backward_configs():
    # Construct Config objects once, outside torch.compile tracing. Width-
    # dependent BLOCK_SIZE_C is supplied by the caller as a constexpr.
    return [triton.Config(dict(BLOCK_SIZE_M=4,STEP_SIZE_C=sc),num_warps=w,num_stages=s)
            for sc,w,s in itertools.product((64,128),(1,2),(2,3))]


@triton.autotune(configs=_forward_configs(),key=['M','C','HAS_POST_SCALE'])
@triton.jit
def _nvidia_scaled_fwd(
    f_ptr,  # (M, C)
    bias_ptr,  # (C,), or None if HAS_BIAS is False
    H_post_ptr,  # (M, n)
    x_ptr,  # (M, C, n)
    H_res_ptr,  # (M, n, n)
    a_ptr,  # Added static FP32 channel/input-stream scale (C,n)
    post_scale_ptr,  # Optional static (C,n) branch-to-stream scale
    output_ptr,  # # (M, C, n)
    M,
    C,
    n: tl.constexpr,
    stride_fm,
    stride_fc,
    stride_bias,  # Not used if HAS_BIAS is False
    stride_xm,
    stride_xCn,
    stride_output_m,
    stride_output_Cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_POST_SCALE: tl.constexpr,
):
    """
    output = (f + bias[None, :, None]) @ H_post: (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n)  = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
           + x @ H_res: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    """
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(0)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_fm > 0 and stride_fc == 1)
    tl.assume(stride_bias == 1)
    tl.assume(stride_xm > 0 and stride_xCn == 1)
    tl.assume(stride_output_m > 0 and stride_output_Cn == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_cn = pid_c * BLOCK_SIZE_C * n + tl.arange(0, BLOCK_SIZE_C * n)
    mask_m = offs_m < M
    mask_c = offs_c < C
    mask_cn = offs_cn < C * n

    f_ptrs = f_ptr + offs_m[:, None] * stride_fm + offs_c[None, :] * stride_fc
    f = tl.load(f_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_c * stride_bias, mask=mask_c, other=0.0)  # (BLOCK_SIZE_C,)

    offs_H_post = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_post = tl.load(
        H_post_ptr + offs_H_post, mask=offs_H_post < M * n, other=0.0, cache_modifier=".ca"
    )
    H_post = tl.reshape(H_post, (BLOCK_SIZE_M, n))  # (BLOCK_SIZE_M, n)

    # Residual connection path: res_out = f @ H_post + bias @ H_post:
    # (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n)  = (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
    # Due to broadcasting, it's equivalent to a multiplicaiton
    out_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C, n), dtype=tl.float32)
    if HAS_BIAS:
        out_acc = tl.fma(bias[None, :, None], H_post[:, None, :], out_acc)

    H_res_offs = pid_m * BLOCK_SIZE_M * n * n + tl.arange(0, BLOCK_SIZE_M * n * n)
    H_res = tl.load(
        H_res_ptr + H_res_offs, mask=H_res_offs < M * n * n, other=0.0, cache_modifier=".ca"
    )
    H_res = tl.reshape(H_res, (BLOCK_SIZE_M, n, n))  # (BLOCK_SIZE_M, n, n)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
    x = tl.load(
        x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, n)

    # Added modulation, staying in FP32: no new BF16 rounding of x*a.
    a = tl.load(a_ptr + offs_cn, mask=mask_cn, other=0.0)
    x = x.to(tl.float32) * a[None, :]

    # Manifold connection path: manifold_out = H_res @ x:
    # (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    # triton doesn't support dot prod with inner dimension < 16, so we need to manually unroll the computation for n=4:
    # x @ H_res = x[:, :, 0] @ H_res[:, 0, :]
    #           + x[:, :, 1] @ H_res[:, 1, :]
    #           + x[:, :, 2] @ H_res[:, 2, :]
    #           + x[:, :, 3] @ H_res[:, 3, :]

    x_reshape = tl.reshape(x, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2))
    x01, x23 = tl.split(
        x_reshape
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, 2), (BLOCK_SIZE_M, BLOCK_SIZE_C, 2)
    x0, x1 = tl.split(x01)  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)
    x2, x3 = tl.split(x23)  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)

    H_resT = tl.reshape(tl.trans(H_res, (0, 2, 1)), (BLOCK_SIZE_M, n, 2, 2))
    H_res01, H_res23 = tl.split(H_resT)  # (BLOCK_SIZE_M, n, 2), (BLOCK_SIZE_M, n, 2)
    H_res0, H_res1 = tl.split(H_res01)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)
    H_res2, H_res3 = tl.split(H_res23)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)

    out_acc = tl.fma(x0[:, :, None], H_res0[:, None, :], out_acc)
    out_acc = tl.fma(x1[:, :, None], H_res1[:, None, :], out_acc)
    out_acc = tl.fma(x2[:, :, None], H_res2[:, None, :], out_acc)
    out_acc = tl.fma(x3[:, :, None], H_res3[:, None, :], out_acc)

    # Preserve the accepted scale implementation's branch-last FMA order.
    if HAS_POST_SCALE:
        post_scale = tl.load(post_scale_ptr + offs_cn, mask=mask_cn, other=0.0).reshape(BLOCK_SIZE_C,n)
        branch = f.to(tl.float32)[:, :, None] * post_scale[None, :, :]
        out_acc = tl.fma(branch, H_post[:, None, :], out_acc)
    else:
        out_acc = tl.fma(f[:, :, None], H_post[:, None, :], out_acc)
    out = out_acc.to(output_ptr.dtype.element_ty)
    out = tl.reshape(out, (BLOCK_SIZE_M, BLOCK_SIZE_C * n))  # (BLOCK_SIZE_M, BLOCK_SIZE_C*n)

    output_ptrs = (
        output_ptr + offs_m[:, None] * stride_output_m + offs_cn[None, :] * stride_output_Cn
    )
    tl.store(output_ptrs, out, mask=mask_m[:, None] & mask_cn[None, :])


@triton.autotune(configs=_backward_configs(),key=["M","C","HAS_POST_SCALE"])
@triton.jit
def _nvidia_scaled_bwd(
    grad_output_ptr,  # (M, C, n)
    f_ptr,  # (M, C)
    bias_ptr,  # (C,), or None if HAS_BIAS is False
    H_post_ptr,  # (M, n)
    x_ptr,  # (M, C, n)
    H_res_ptr,  # (M, n, n)
    grad_H_post_ptr,  # (M, n)
    grad_f_ptr,  # (M, C)
    grad_bias_ptr,  # (C,), or None if HAS_BIAS is False
    grad_bias_ws_ptr,  # (grid_m, C), or None if HAS_BIAS is False or DETERMINISTIC is False
    grad_H_res_ptr,  # (M, n, n)
    grad_x_ptr,  # (M, C, n)
    a_ptr,  # Added static FP32 input scale (C,n)
    grad_a_ws_ptr,  # Added deterministic partials (ceil(M/4),C,n)
    post_scale_ptr,
    grad_post_scale_ws_ptr,
    M,
    C,
    n: tl.constexpr,
    stride_grad_output_m,
    stride_grad_output_Cn,
    stride_fm,
    stride_fc,
    stride_bias,  # Not used if HAS_BIAS is False
    stride_xm,
    stride_xCn,
    stride_grad_fm,
    stride_grad_fc,
    stride_grad_bias,  # Not used if HAS_BIAS is False
    stride_grad_bias_ws_m,  # Not used if HAS_BIAS is False or DETERMINISTIC is False
    stride_grad_bias_ws_c,  # Not used if HAS_BIAS is False or DETERMINISTIC is False
    stride_grad_xm,
    stride_grad_xCn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    STEP_SIZE_C: tl.constexpr,
    precision: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_POST_SCALE: tl.constexpr,
    FUSE_GRAD_X_ACC: tl.constexpr,
    DETERMINISTIC: tl.constexpr,  # If True, grad_bias partials go to a workspace (reduced in the wrapper) instead of atomic_add.
):
    """
    Each block
    It reads
    - (BLOCK_SIZE_M, BLOCK_SIZE_C) of f, which is the output of the attention / FFN module
    - (BLOCK_SIZE_M, n) of H_post, which is applied for the transformation of the attention / FFN output
    - (BLOCK_SIZE_M, BLOCK_SIZE_C, n) of x, which is the skip connection's input
    - (BLOCK_SIZE_M, n*n) of H_res, which is applied for the transformation of the skip connection
    and writes
    - (BLOCK_SIZE_M, n) of grad_H_post
    - (BLOCK_SIZE_M, BLOCK_SIZE_C) of grad_f
    - (BLOCK_SIZE_M, n, n) of grad_H_res
    - (BLOCK_SIZE_M, BLOCK_SIZE_C, n) of grad_x

    Forward:
        out = f @ H_post + x @ H_res
    Backward:
        GEMM:
        grad_H_post = f.T @ grad_output: (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, 1, n)
        grad_H_res = x.T @ grad_output: (BLOCK_SIZE_M, n, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, n, n)
        Not GEMM:
        grad_f = grad_output @ H_post.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, 1) = (BLOCK_SIZE_M, BLOCK_SIZE_C, 1)
        grad_x = grad_output @ H_res.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    """

    pid_m = tl.program_id(1)
    pid_c = tl.program_id(0)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_fm > 0 and stride_fc == 1)
    tl.assume(stride_bias == 1)
    tl.assume(stride_xm > 0 and stride_xCn == 1)
    tl.assume(stride_grad_output_m > 0 and stride_grad_output_Cn == 1)
    tl.assume(stride_grad_fm > 0 and stride_grad_fc == 1)
    tl.assume(stride_grad_bias == 1)
    tl.assume(stride_grad_xm > 0 and stride_grad_xCn == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M

    offs_c_start = pid_c * BLOCK_SIZE_C
    offs_cn_start = pid_c * BLOCK_SIZE_C * n

    grad_H_post_acc = tl.zeros((BLOCK_SIZE_M, 1, n), dtype=tl.float32)
    grad_H_res_acc = tl.zeros((BLOCK_SIZE_M, n, n), dtype=tl.float32)

    H_post_offs = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_post = tl.load(H_post_ptr + H_post_offs, mask=H_post_offs < M * n, other=0.0)
    H_post_reshape = tl.reshape(H_post, (BLOCK_SIZE_M, 2, 2))
    H_post01, H_post23 = tl.split(H_post_reshape)  # (BLOCK_SIZE_M, 2), (BLOCK_SIZE_M, 2)
    H_post0, H_post1 = tl.split(H_post01)  # (BLOCK_SIZE_M,), (BLOCK_SIZE_M,)
    H_post2, H_post3 = tl.split(H_post23)  # (BLOCK_SIZE_M,), (BLOCK_SIZE_M,)

    H_res_offs = pid_m * BLOCK_SIZE_M * n * n + tl.arange(0, BLOCK_SIZE_M * n * n)
    H_res = tl.load(
        H_res_ptr + H_res_offs, mask=H_res_offs < M * n * n, other=0.0
    )  # (BLOCK_SIZE_M, n, n)
    H_res = tl.reshape(H_res, (BLOCK_SIZE_M, n, n))  # (BLOCK_SIZE_M, n, n)
    H_res_reshape = tl.reshape(H_res, (BLOCK_SIZE_M, n, 2, 2))  # (BLOCK_SIZE_M, n, 2, 2)
    H_res01, H_res23 = tl.split(H_res_reshape)  # (BLOCK_SIZE_M, n, 2), (BLOCK_SIZE_M, n, 2)
    H_res0, H_res1 = tl.split(H_res01)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)
    H_res2, H_res3 = tl.split(H_res23)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)

    for i in tl.range(0, BLOCK_SIZE_C, STEP_SIZE_C, loop_unroll_factor=2):
        offs_c = offs_c_start + i + tl.arange(0, STEP_SIZE_C)
        offs_cn = offs_cn_start + i * n + tl.arange(0, STEP_SIZE_C * n)
        mask_c = offs_c < C
        mask_cn = offs_cn < C * n

        f_ptrs = f_ptr + offs_m[:, None] * stride_fm + offs_c[None, :] * stride_fc
        f = tl.load(f_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0)

        if HAS_BIAS:
            bias = tl.load(
                bias_ptr + offs_c * stride_bias, mask=mask_c, other=0.0
            )  # (STEP_SIZE_C,)

        grad_out_ptrs = (
            grad_output_ptr
            + offs_m[:, None] * stride_grad_output_m
            + offs_cn[None, :] * stride_grad_output_Cn
        )
        grad_out = tl.load(
            grad_out_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
        )  # (BLOCK_SIZE_M, STEP_SIZE_C * n)
        grad_out = tl.reshape(
            grad_out, (BLOCK_SIZE_M, STEP_SIZE_C, n)
        )  # (BLOCK_SIZE_M, STEP_SIZE_C, n)

        # grad_H_post =  f.T @ grad_output # (BLOCK_SIZE_M, 1, STEP_SIZE_C) @ (BLOCK_SIZE_M, STEP_SIZE_C, n) = (BLOCK_SIZE_M, 1, n)
        if HAS_POST_SCALE:
            post_scale = tl.load(post_scale_ptr + offs_cn, mask=mask_cn, other=0.0).reshape(STEP_SIZE_C,n)
            scaled_grad_out = grad_out.to(tl.float32) * post_scale[None,:,:]
            grad_H_post_acc = tl.dot(
                f.to(tl.float32).reshape(BLOCK_SIZE_M,1,STEP_SIZE_C),
                scaled_grad_out, acc=grad_H_post_acc, input_precision="ieee", out_dtype=tl.float32)
            post_part = tl.sum(grad_out.to(tl.float32) * f.to(tl.float32)[:,:,None] * H_post.reshape(BLOCK_SIZE_M,n)[:,None,:], axis=0)
            tl.store(grad_post_scale_ws_ptr + pid_m*C*n + offs_cn,post_part.reshape(STEP_SIZE_C*n),mask=mask_cn)
        else:
            grad_H_post_acc = tl.dot(
                tl.reshape(f, (BLOCK_SIZE_M, 1, STEP_SIZE_C)),
                tl.reshape(grad_out, (BLOCK_SIZE_M, STEP_SIZE_C, n)),
                acc=grad_H_post_acc,
                input_precision=precision,
                out_dtype=tl.float32,
            )  # (BLOCK_SIZE_M, 1, n)
        if HAS_BIAS:
            grad_H_post_acc = tl.dot(
                tl.broadcast_to(bias[None, None, :], (BLOCK_SIZE_M, 1, STEP_SIZE_C)),
                tl.reshape(grad_out, (BLOCK_SIZE_M, STEP_SIZE_C, n)),
                acc=grad_H_post_acc,
                input_precision=precision,
                out_dtype=tl.float32,
            )  # (BLOCK_SIZE_M, 1, n)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
        x = tl.load(
            x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
        )  # (BLOCK_SIZE_M, STEP_SIZE_C*n)
        x = tl.reshape(x, (BLOCK_SIZE_M, STEP_SIZE_C, n))  # (BLOCK_SIZE_M, STEP_SIZE_C, n)

        # Added modulation inside NVIDIA's existing read/compute/write loop.
        a = tl.load(a_ptr + offs_cn, mask=mask_cn, other=0.0)
        a = tl.reshape(a, (STEP_SIZE_C, n))
        scaled_x = x.to(tl.float32) * a[None, :, :]

        # grad_H_res = x.T @ grad_output: (BLOCK_SIZE_M, n, STEP_SIZE_C) @ (BLOCK_SIZE_M, STEP_SIZE_C, n) = (BLOCK_SIZE_M, n, n)
        grad_H_res_acc = tl.dot(
            tl.trans(scaled_x, (0, 2, 1)),
            grad_out.to(tl.float32),
            acc=grad_H_res_acc,
            # Preserve FP32 scaled-input precision, including scale near zero.
            input_precision="ieee",
            out_dtype=tl.float32,
        )  # (BLOCK_SIZE_M, n, n)

        grad_out_reshape = tl.reshape(
            grad_out, (BLOCK_SIZE_M, STEP_SIZE_C, 2, 2)
        )  # (BLOCK_SIZE_M, STEP_SIZE_C, 2, 2)
        grad_out01, grad_out23 = tl.split(
            grad_out_reshape
        )  # (BLOCK_SIZE_M, STEP_SIZE_C, 2), (BLOCK_SIZE_M, STEP_SIZE_C, 2)
        grad_out0, grad_out1 = tl.split(
            grad_out01
        )  # (BLOCK_SIZE_M, STEP_SIZE_C), (BLOCK_SIZE_M, STEP_SIZE_C)
        grad_out2, grad_out3 = tl.split(
            grad_out23
        )  # (BLOCK_SIZE_M, STEP_SIZE_C), (BLOCK_SIZE_M, STEP_SIZE_C)

        # grad_f = grad_output @ H_post.T: (BLOCK_SIZE_M, 1, n) @ (BLOCK_SIZE_M, n, STEP_SIZE_C) = (BLOCK_SIZE_M, 1, STEP_SIZE_C)
        # Triton doesn't support dot prod with inner dimension < 16, so we need to hack this:
        #        = grad_out[:, :, 0] @ H_post.T[:, 0, :] (BLOCK_SIZE_M, STEP_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, 1)
        #        + grad_out[:, :, 1] @ H_post.T[:, 1, :]
        #        + grad_out[:, :, 2] @ H_post.T[:, 2, :]
        #        + grad_out[:, :, 3] @ H_post.T[:, 3, :]
        # where H_post.T[:, i, :] = H_post[:, :, i]

        if HAS_POST_SCALE:
            post_scale01,post_scale23 = tl.split(post_scale.reshape(STEP_SIZE_C,2,2))
            post_scale0,post_scale1 = tl.split(post_scale01)
            post_scale2,post_scale3 = tl.split(post_scale23)
            grad_f_acc = tl.zeros((BLOCK_SIZE_M, STEP_SIZE_C),dtype=tl.float32)
            grad_f_acc = tl.fma(grad_out0,H_post0[:,None]*post_scale0[None,:],grad_f_acc)
            grad_f_acc = tl.fma(grad_out1,H_post1[:,None]*post_scale1[None,:],grad_f_acc)
            grad_f_acc = tl.fma(grad_out2,H_post2[:,None]*post_scale2[None,:],grad_f_acc)
            grad_f_acc = tl.fma(grad_out3,H_post3[:,None]*post_scale3[None,:],grad_f_acc)
        else:
            grad_f_acc = tl.zeros((BLOCK_SIZE_M, STEP_SIZE_C), dtype=tl.float32)
            # (BLOCK_SIZE_M, STEP_SIZE_C) * (BLOCK_SIZE_M, 1) -> (BLOCK_SIZE_M, STEP_SIZE_C)
            grad_f_acc = tl.fma(grad_out0, H_post0[:, None], grad_f_acc)
            grad_f_acc = tl.fma(grad_out1, H_post1[:, None], grad_f_acc)
            grad_f_acc = tl.fma(grad_out2, H_post2[:, None], grad_f_acc)
            grad_f_acc = tl.fma(grad_out3, H_post3[:, None], grad_f_acc)
        grad_f = grad_f_acc.to(f.dtype)

        grad_f_ptrs = (
            grad_f_ptr + offs_m[:, None] * stride_grad_fm + offs_c[None, :] * stride_grad_fc
        )
        tl.store(grad_f_ptrs, grad_f, mask=mask_m[:, None] & mask_c[None, :])

        if HAS_BIAS:
            grad_bias = tl.sum(grad_f_acc, axis=0)  # (STEP_SIZE_C,)
            # This is reduction over M dimension, so it has nothing to do with whether we use split-C. It only depends on determinism or not.
            if DETERMINISTIC:
                grad_bias_ws_ptrs = (
                    grad_bias_ws_ptr
                    + pid_m * stride_grad_bias_ws_m
                    + offs_c * stride_grad_bias_ws_c
                )
                tl.store(grad_bias_ws_ptrs, grad_bias, mask=mask_c)
            else:
                grad_bias_ptrs = grad_bias_ptr + offs_c * stride_grad_bias
                tl.atomic_add(grad_bias_ptrs, grad_bias, mask=mask_c, sem="relaxed")

        # grad_x = grad_output @ H_res.T: (BLOCK_SIZE_M, STEP_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, n, STEP_SIZE_C)
        # The inner dim is n=4 which is too small for triton, so we will manually unroll the matmul
        # grad_x = grad_out[:, :, 0] @ H_res.T[:, 0, :]
        #        + grad_out[:, :, 1] @ H_res.T[:, 1, :]
        #        + grad_out[:, :, 2] @ H_res.T[:, 2, :]
        #        + grad_out[:, :, 3] @ H_res.T[:, 3, :]
        # where H_res.T[:, i, :] = H_res[:, :, i]
        # Due to broadcasting, it's equivalent to multiplying each H_res[:, i, :].T with grad_out[:, i, :]

        grad_x_acc = tl.zeros((BLOCK_SIZE_M, STEP_SIZE_C, n), dtype=tl.float32)
        grad_x_acc = tl.fma(grad_out0[:, :, None], H_res0[:, None, :], grad_x_acc)
        grad_x_acc = tl.fma(grad_out1[:, :, None], H_res1[:, None, :], grad_x_acc)
        grad_x_acc = tl.fma(grad_out2[:, :, None], H_res2[:, None, :], grad_x_acc)
        grad_x_acc = tl.fma(grad_out3[:, :, None], H_res3[:, None, :], grad_x_acc)

        # grad_x_acc is dL/d(x*a), before multiplying by a.
        # Sum this CTA's token rows; the small final kernel sums CTA partials.
        grad_a = tl.sum(grad_x_acc * x.to(tl.float32), axis=0)
        tl.store(grad_a_ws_ptr + pid_m*C*n + offs_cn,
                 tl.reshape(grad_a, (STEP_SIZE_C*n,)), mask=mask_cn)
        grad_x_acc *= a[None, :, :]

        if FUSE_GRAD_X_ACC:
            grad_x = grad_x_acc  # If fusing gradient accumulation, the buffer should be always fp32 so we don't cast here
        else:
            grad_x = grad_x_acc.to(x.dtype)
        grad_x = tl.reshape(
            grad_x, (BLOCK_SIZE_M, STEP_SIZE_C * n)
        )  # (BLOCK_SIZE_M, STEP_SIZE_C*n)

        grad_x_ptrs = (
            grad_x_ptr + offs_m[:, None] * stride_grad_xm + offs_cn[None, :] * stride_grad_xCn
        )
        tl.store(grad_x_ptrs, grad_x, mask=mask_m[:, None] & mask_cn[None, :])

    grad_H_post = tl.reshape(grad_H_post_acc, (BLOCK_SIZE_M * n,))  # (BLOCK_SIZE_M * n)
    offs_grad_H_post = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    grad_H_post_ptrs = grad_H_post_ptr + offs_grad_H_post

    grad_H_res = tl.reshape(grad_H_res_acc, (BLOCK_SIZE_M * n * n,))  # (BLOCK_SIZE_M * n * n)
    offs_grad_H_res = pid_m * BLOCK_SIZE_M * n * n + tl.arange(0, BLOCK_SIZE_M * n * n)
    grad_H_res_ptrs = grad_H_res_ptr + offs_grad_H_res

    # A single C block covers the reduction, so no atomic_add is needed.
    tl.store(grad_H_post_ptrs, grad_H_post.to(H_post.dtype), mask=offs_grad_H_post < M * n)
    tl.store(grad_H_res_ptrs, grad_H_res.to(H_res.dtype), mask=offs_grad_H_res < M * n * n)
