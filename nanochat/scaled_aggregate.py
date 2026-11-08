# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See research/connections/scaled-hc-nvidia/LICENSE.NVIDIA.
# Adapted from TE2.20.2 aggregate: FP32 source scale and deterministic scale VJP.
import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from nanochat.scaled_nvidia_kernels import _forward_configs
from nanochat.scaled_reduction import reduce_scale_partials


@triton.autotune(configs=_forward_configs(), key=['M','C'])
@triton.jit
def _scaled_aggregate_fwd(
    x_ptr,  # # (M, C, n)
    H_pre_ptr,  # (M, n)
    output_ptr,  # (M, C)
    a_ptr,  # (C,n), FP32 static channel/source scale
    M,
    C,
    n: tl.constexpr,
    stride_xm,
    stride_xCn,
    stride_output_m,
    stride_output_c,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    output = x @ H_pre: (M, C, n) @ (M, n, 1) = (M, C, 1)
    """
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(0)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_xm > 0 and stride_xCn == 1)
    tl.assume(stride_output_m > 0 and stride_output_c == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_cn = pid_c * BLOCK_SIZE_C * n + tl.arange(0, BLOCK_SIZE_C * n)
    mask_m = offs_m < M
    mask_c = offs_c < C
    mask_cn = offs_cn < C * n

    offs_H_pre = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_pre = tl.load(
        H_pre_ptr + offs_H_pre, mask=offs_H_pre < M * n, other=0.0, cache_modifier=".ca"
    )  # (BLOCK_SIZE_M * n)
    H_pre = H_pre.reshape(BLOCK_SIZE_M, 2, 2)
    H_pre01, H_pre23 = tl.split(H_pre)
    H_pre0, H_pre1 = tl.split(H_pre01)
    H_pre2, H_pre3 = tl.split(H_pre23)  # (BLOCK_SIZE_M, 1)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
    x = tl.load(
        x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C * n)

    a = tl.load(a_ptr + offs_cn, mask=mask_cn, other=0.0)
    x = x.to(tl.float32) * a[None, :]
    x = tl.reshape(x, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2))
    x01, x23 = tl.split(x)
    x0, x1 = tl.split(x01)
    x2, x3 = tl.split(x23)  # (BLOCK_SIZE_M, BLOCK_SIZE_C)

    # x @ H_pre: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, 1)
    # triton doesn't support dot prod with inner dimension < 16, so we need to manually unroll the computation for n=4:
    # x @ H_pre = x[:, :, 0] * H_pre[:, 0]
    #           + x[:, :, 1] * H_pre[:, 1]
    #           + x[:, :, 2] * H_pre[:, 2]
    #           + x[:, :, 3] * H_pre[:, 3]
    out_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C), dtype=tl.float32)
    out_acc = tl.fma(x0, H_pre0[:, None], out_acc)
    out_acc = tl.fma(x1, H_pre1[:, None], out_acc)
    out_acc = tl.fma(x2, H_pre2[:, None], out_acc)
    out_acc = tl.fma(x3, H_pre3[:, None], out_acc)

    out = out_acc.to(output_ptr.dtype.element_ty)

    output_ptrs = output_ptr + offs_m[:, None] * stride_output_m + offs_c[None, :] * stride_output_c
    tl.store(output_ptrs, out, mask=mask_m[:, None] & mask_c[None, :])


@triton.autotune(configs=[triton.Config(dict(BLOCK_SIZE_M=4, STEP_SIZE_C=64), num_warps=1, num_stages=s) for s in (2,3,4)], key=['M','C'])
@triton.jit
def _scaled_aggregate_bwd(
    grad_output_ptr,  # (M, C)
    H_pre_ptr,  # (M, n)
    grad_H_pre_ptr,  # (M, n)
    x_ptr,  # (M, C, n)
    grad_x_ptr,  # # (M, C, n)
    a_ptr,
    grad_a_ws_ptr,
    M,
    C,
    n: tl.constexpr,
    stride_grad_output_m,
    stride_grad_output_c,
    stride_xm,
    stride_xCn,
    stride_grad_xm,
    stride_grad_xCn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    STEP_SIZE_C: tl.constexpr,
    precision: tl.constexpr,
    FUSE_GRAD_X_ACC: tl.constexpr,
):
    """
    Forward:
        out = x @ H_pre: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, 1) = (BLOCK_SIZE_M, BLOCK_SIZE_C, 1)
    Backward:
        grad_H_pre = x.T @ grad_output: (BLOCK_SIZE_M, n, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) = (BLOCK_SIZE_M, n, 1)
        grad_H_pre.T = grad_output.T @ x: (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, 1, n)
            which is easier to compute since transposing grad_H_pre and grad_output is just view change
        grad_x = grad_output @ H_pre.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    """
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(0)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_xm > 0 and stride_xCn == 1)
    tl.assume(stride_grad_xm > 0 and stride_grad_xCn == 1)
    tl.assume(stride_grad_output_m > 0 and stride_grad_output_c == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)
    tl.assume(STEP_SIZE_C % 32 == 0)
    tl.assume(BLOCK_SIZE_C % STEP_SIZE_C == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M

    offs_c_start = pid_c * BLOCK_SIZE_C
    offs_cn_start = pid_c * BLOCK_SIZE_C * n

    H_pre_offs = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_pre = tl.load(
        H_pre_ptr + H_pre_offs, mask=H_pre_offs < M * n, other=0.0, cache_modifier=".ca"
    )  # (BLOCK_SIZE_M * n)
    H_pre = tl.reshape(H_pre, (BLOCK_SIZE_M, n))  # (BLOCK_SIZE_M, n)

    grad_H_pre_acc = tl.zeros((BLOCK_SIZE_M, 1, n), dtype=tl.float32)
    for i in tl.range(0, BLOCK_SIZE_C, STEP_SIZE_C, loop_unroll_factor=2):
        offs_c = offs_c_start + i + tl.arange(0, STEP_SIZE_C)
        offs_cn = offs_cn_start + i * n + tl.arange(0, STEP_SIZE_C * n)
        mask_c = offs_c < C
        mask_cn = offs_cn < C * n

        grad_output_ptrs = (
            grad_output_ptr
            + offs_m[:, None] * stride_grad_output_m
            + offs_c[None, :] * stride_grad_output_c
        )
        grad_output = tl.load(
            grad_output_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0
        )  # (BLOCK_SIZE_M, STEP_SIZE_C)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
        x = tl.load(
            x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
        )  # (BLOCK_SIZE_M, STEP_SIZE_C * n)

        a = tl.load(a_ptr + offs_cn, mask=mask_cn, other=0.0).reshape(STEP_SIZE_C, n)
        scaled_x = x.reshape(BLOCK_SIZE_M, STEP_SIZE_C, n).to(tl.float32) * a[None, :, :]
        grad_H_pre_acc = tl.dot(
            tl.reshape(grad_output.to(tl.float32), (BLOCK_SIZE_M, 1, STEP_SIZE_C)),
            scaled_x,
            acc=grad_H_pre_acc,
            input_precision=precision,
            out_dtype=tl.float32,
        )

        # grad_x = grad_output @ H_pre.T: (BLOCK_SIZE_M, STEP_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n) = (BLOCK_SIZE_M, STEP_SIZE_C, n)
        grad_x = grad_output[:, :, None] * H_pre[:, None, :]  # (BLOCK_SIZE_M, STEP_SIZE_C, n)
        grad_a = tl.sum(grad_x * x.reshape(BLOCK_SIZE_M, STEP_SIZE_C, n).to(tl.float32), axis=0)
        tl.store(grad_a_ws_ptr + pid_m*C*n + offs_cn, grad_a.reshape(STEP_SIZE_C*n), mask=mask_cn)
        grad_x *= a[None, :, :]
        grad_x = tl.reshape(grad_x, (BLOCK_SIZE_M, STEP_SIZE_C * n))

        grad_x_ptrs = (
            grad_x_ptr + offs_m[:, None] * stride_grad_xm + offs_cn[None, :] * stride_grad_xCn
        )

        if FUSE_GRAD_X_ACC:  # If fused gradient accumulation is enabled, the buffer is always fp32
            grad_x_acc = tl.load(grad_x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0)
            grad_x = grad_x.to(tl.float32) + grad_x_acc
        tl.store(
            grad_x_ptrs,
            grad_x,
            mask=mask_m[:, None] & mask_cn[None, :],
        )

    grad_H_pre = tl.reshape(grad_H_pre_acc, (BLOCK_SIZE_M * n,))  # (BLOCK_SIZE_M * n)
    offs_grad_H_pre = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    grad_H_pre_ptrs = grad_H_pre_ptr + offs_grad_H_pre
    # A single C block covers the reduction, so no atomic_add is needed.
    tl.store(grad_H_pre_ptrs, grad_H_pre.to(H_pre.dtype), mask=offs_grad_H_pre < M * n)


class _ScaledAggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,pre,a):
        if x.ndim!=4 or x.shape[-1]!=4 or not x.is_cuda or x.dtype!=torch.bfloat16:
            raise ValueError('Scaled aggregate expects CUDA BF16 [B,T,C,4]')
        c=x.shape[-2];m=x.numel()//(c*4)
        if m<1 or pre.shape!=(*x.shape[:2],4) or a.shape!=(c,4):
            raise ValueError('Invalid aggregate routing/scale shapes')
        if any(p.dtype!=torch.float32 for p in (pre,a)):
            raise ValueError('Aggregate routing and scale must be FP32')
        if not all(p.is_contiguous() and p.device==x.device for p in (x,pre,a)):
            raise ValueError('Scaled aggregate inputs must be contiguous and colocated')
        out=torch.empty(x.shape[:-1],device=x.device,dtype=x.dtype)
        _scaled_aggregate_fwd[lambda meta:(triton.cdiv(c,meta['BLOCK_SIZE_C']),triton.cdiv(m,meta['BLOCK_SIZE_M']))](
            x,pre,out,a,m,c,4,c*4,1,c,1)
        ctx.save_for_backward(x,pre,a)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,go):
        x,pre,a=ctx.saved_tensors;go=go.contiguous()
        c=x.shape[-2];m=x.numel()//(c*4)
        gx=torch.empty_like(x);gp=torch.empty_like(pre)
        parts=torch.empty((1,triton.cdiv(m,4),c,4),device=x.device,dtype=torch.float32)
        _scaled_aggregate_bwd[lambda meta:(triton.cdiv(c,meta['BLOCK_SIZE_C']),triton.cdiv(m,meta['BLOCK_SIZE_M']))](
            go,pre,gp,x,gx,a,parts,m,c,4,c,1,c*4,1,c*4,1,
            BLOCK_SIZE_C=triton.cdiv(c,64)*64,precision='ieee',FUSE_GRAD_X_ACC=False)
        return gx,gp,reduce_scale_partials(parts)[0]


def scaled_aggregate(x,pre,a):
    return _ScaledAggregate.apply(x,pre,a)
