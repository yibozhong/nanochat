"""Static channel/input-stream scales in NVIDIA's fused HC write kernels.

Formula/parameters are unchanged from the first validated scaled_hc version.
BF16 states; FP32 scale, scaled input, routing arithmetic and reduction.
Forward and the main backward each scan activations once; only parameter
partials are written for the final deterministic scale-gradient reduction.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from nanochat.scaled_nvidia_kernels import _nvidia_scaled_fwd, _nvidia_scaled_bwd


@tr.jit
def _scaled_da_reduce(Parts, GA, K: tl.constexpr, S: tl.constexpr):
    # Stream the64MiB workspace in bounded tiles, not one8192-row tensor.
    k=tl.program_id(0)*16+tl.arange(0,16)
    row=tl.arange(0,128)
    acc=tl.full((128,16),0.,tl.float32)
    for start in range(0,S,128):
        s=start+row
        acc+=tl.load(Parts+s[:,None]*K+k[None,:],(s[:,None]<S)&(k[None,:]<K),0)
    tl.store(GA+k,tl.sum(acc,0),k<K)


class _ScaledMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, post, h, scale):
        if x.ndim != 4 or x.shape[-1] != 4 or not x.is_cuda:
            raise ValueError('Scaled HC expects CUDA [batch,time,channel,4]')
        c = x.shape[-2]; n = x.numel()//(4*c)
        if n < 1 or x.dtype != torch.bfloat16 or y.dtype != x.dtype:
            raise ValueError('Scaled HC uses nonempty BF16 states')
        if y.shape != x.shape[:-1] or post.shape != (*x.shape[:2],4) or h.shape != (*x.shape[:2],4,4) or scale.shape != (c,4):
            raise ValueError('Invalid branch/routing/scale shapes')
        if any(p.dtype != torch.float32 for p in (post,h,scale)):
            raise ValueError('Routing and scale must be FP32')
        if not all(p.is_contiguous() and p.device == x.device for p in (x,y,post,h,scale)):
            raise ValueError('Scaled HC requires contiguous colocated inputs')
        out = torch.empty_like(x)
        _nvidia_scaled_fwd[lambda meta: (tr.cdiv(c,meta['BLOCK_SIZE_C']),tr.cdiv(n,meta['BLOCK_SIZE_M']))](
            f_ptr=y,bias_ptr=y,H_post_ptr=post,x_ptr=x,H_res_ptr=h,a_ptr=scale,output_ptr=out,
            M=n,C=c,n=4,stride_fm=c,stride_fc=1,stride_bias=1,stride_xm=c*4,stride_xCn=1,
            stride_output_m=c*4,stride_output_Cn=1,HAS_BIAS=False)
        ctx.save_for_backward(x,y,post,h,scale)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, go):
        x,y,post,h,scale = ctx.saved_tensors
        go=go.contiguous(); c=x.shape[-2]; n=x.numel()//(4*c)
        gx=torch.empty_like(x);gy=torch.empty_like(y)
        gp=torch.empty_like(post);gh=torch.empty_like(h);ga=torch.empty_like(scale)
        # NVIDIA owns4 tokens per CTA and traverses all channels once.
        # 64MiB deterministic parameter partials at N32768/C512; not a saved state.
        splits=tr.cdiv(n,4)
        parts=torch.empty((splits,c,4),device=x.device,dtype=torch.float32)
        _nvidia_scaled_bwd[lambda meta: (tr.cdiv(c,meta['BLOCK_SIZE_C']),tr.cdiv(n,meta['BLOCK_SIZE_M']))](
            grad_output_ptr=go,f_ptr=y,bias_ptr=y,H_post_ptr=post,x_ptr=x,H_res_ptr=h,
            grad_H_post_ptr=gp,grad_f_ptr=gy,grad_bias_ptr=ga,grad_bias_ws_ptr=parts,
            grad_H_res_ptr=gh,grad_x_ptr=gx,a_ptr=scale,grad_a_ws_ptr=parts,M=n,C=c,n=4,
            stride_grad_output_m=c*4,stride_grad_output_Cn=1,stride_fm=c,stride_fc=1,stride_bias=1,
            stride_xm=c*4,stride_xCn=1,stride_grad_fm=c,stride_grad_fc=1,stride_grad_bias=1,
            stride_grad_bias_ws_m=c,stride_grad_bias_ws_c=1,stride_grad_xm=c*4,stride_grad_xCn=1,
            precision='ieee',HAS_BIAS=False,FUSE_GRAD_X_ACC=False,DETERMINISTIC=True,
            BLOCK_SIZE_C=tr.cdiv(c,32)*32)
        _scaled_da_reduce[(tr.cdiv(c*4,16),)](parts,ga,c*4,splits,num_warps=4)
        return gx,gy,gp,gh,ga


def scaled_mix(x, y, post, h, scale):
    return _ScaledMix.apply(x,y,post,h,scale)
