"""Fused alternatives for the shared 4->16->4 stream mixer.

The rowwise path follows NVIDIA TE's expand/combine strategy: split the four
streams and perform local FMAs, without reductions across the stream axis.
The dot path exploits our *shared* weights with matrix products, including
the weight VJPs. Unlike TE's dynamic 4x4 mix, these VJPs reduce across rows.
All hidden intermediates stay on chip and are recomputed during backward.
TF32x3 dots preserve the existing FP32 small-matrix precision policy; no
BF16 quantization of the hidden activation or learned mixer is introduced.

Reference: TransformerEngine v2.20.2 common/triton/mhc.py, expand_combine
forward and backward. This is an independent implementation of our MLP.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from nanochat.channel_mixing import _backward_branch

IMPLEMENTATIONS = ('legacy', 'rowwise', 'hybrid')


@tr.jit
def _rows(X, row, N: tl.constexpr):
    x0 = tl.load(X+row*4, row<N, 0).to(tl.float32)
    x1 = tl.load(X+row*4+1, row<N, 0).to(tl.float32)
    x2 = tl.load(X+row*4+2, row<N, 0).to(tl.float32)
    x3 = tl.load(X+row*4+3, row<N, 0).to(tl.float32)
    return x0, x1, x2, x3


@tr.jit
def _unit(x0, x1, x2, x3, Win, j: tl.constexpr):
    return tl.fma(x3, tl.load(Win+j*4+3), tl.fma(x2, tl.load(Win+j*4+2),
           tl.fma(x1, tl.load(Win+j*4+1), x0*tl.load(Win+j*4))))


@tr.jit
def _row_forward(X, Y, Post, Win, Wout, Out, N: tl.constexpr,
                 C: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    x0,x1,x2,x3 = _rows(X,row,N)
    o0=tl.full((BLOCK,),0.,tl.float32);o1=tl.full((BLOCK,),0.,tl.float32)
    o2=tl.full((BLOCK,),0.,tl.float32);o3=tl.full((BLOCK,),0.,tl.float32)
    for j in tl.static_range(16):
        a=tl.maximum(_unit(x0,x1,x2,x3,Win,j),0.)
        o0=tl.fma(a,tl.load(Wout+j),o0)
        o1=tl.fma(a,tl.load(Wout+16+j),o1)
        o2=tl.fma(a,tl.load(Wout+32+j),o2)
        o3=tl.fma(a,tl.load(Wout+48+j),o3)
    y=tl.load(Y+row,row<N,0).to(tl.float32)
    p0=tl.load(Post+(row//C)*4,row<N,0)
    p1=tl.load(Post+(row//C)*4+1,row<N,0)
    p2=tl.load(Post+(row//C)*4+2,row<N,0)
    p3=tl.load(Post+(row//C)*4+3,row<N,0)
    tl.store(Out+row*4,tl.fma(y,p0,o0),row<N)
    tl.store(Out+row*4+1,tl.fma(y,p1,o1),row<N)
    tl.store(Out+row*4+2,tl.fma(y,p2,o2),row<N)
    tl.store(Out+row*4+3,tl.fma(y,p3,o3),row<N)


@tr.jit
def _row_backward(X, Win, Wout, GO, GX, Parts, N: tl.constexpr, BLOCK: tl.constexpr):
    tile=tl.program_id(0);row=tile*BLOCK+tl.arange(0,BLOCK)
    x0,x1,x2,x3=_rows(X,row,N)
    g0,g1,g2,g3=_rows(GO,row,N)
    d0=tl.full((BLOCK,),0.,tl.float32);d1=tl.full((BLOCK,),0.,tl.float32)
    d2=tl.full((BLOCK,),0.,tl.float32);d3=tl.full((BLOCK,),0.,tl.float32)
    for j in tl.static_range(16):
        z=_unit(x0,x1,x2,x3,Win,j);a=tl.maximum(z,0.)
        da=tl.fma(g3,tl.load(Wout+48+j),tl.fma(g2,tl.load(Wout+32+j),
           tl.fma(g1,tl.load(Wout+16+j),g0*tl.load(Wout+j))))
        dz=tl.where(z>0.,da,0.)
        d0=tl.fma(dz,tl.load(Win+j*4),d0)
        d1=tl.fma(dz,tl.load(Win+j*4+1),d1)
        d2=tl.fma(dz,tl.load(Win+j*4+2),d2)
        d3=tl.fma(dz,tl.load(Win+j*4+3),d3)
        tl.store(Parts+tile*128+j*4,tl.sum(dz*x0,0))
        tl.store(Parts+tile*128+j*4+1,tl.sum(dz*x1,0))
        tl.store(Parts+tile*128+j*4+2,tl.sum(dz*x2,0))
        tl.store(Parts+tile*128+j*4+3,tl.sum(dz*x3,0))
        tl.store(Parts+tile*128+64+j,tl.sum(g0*a,0))
        tl.store(Parts+tile*128+80+j,tl.sum(g1*a,0))
        tl.store(Parts+tile*128+96+j,tl.sum(g2*a,0))
        tl.store(Parts+tile*128+112+j,tl.sum(g3*a,0))
    tl.store(GX+row*4,d0,row<N);tl.store(GX+row*4+1,d1,row<N)
    tl.store(GX+row*4+2,d2,row<N);tl.store(GX+row*4+3,d3,row<N)


@tr.jit
def _dot_hidden(X, Win, row, N: tl.constexpr):
    k=tl.arange(0,16)
    x=tl.load(X+row[:,None]*4+k[None,:],(row[:,None]<N)&(k[None,:]<4),0).to(tl.float32)
    wi=tl.load(Win+k[:,None]*4+k[None,:],k[None,:]<4,0)
    # Use the same four-term FMA order as the rowwise forward for its ReLU mask.
    x0,x1,x2,x3=_rows(X,row,N)
    w0=tl.load(Win+k*4);w1=tl.load(Win+k*4+1)
    w2=tl.load(Win+k*4+2);w3=tl.load(Win+k*4+3)
    z=tl.fma(x3[:,None],w3[None,:],tl.fma(x2[:,None],w2[None,:],
      tl.fma(x1[:,None],w1[None,:],x0[:,None]*w0[None,:])))
    return x,wi,z,tl.maximum(z,0.)


@tr.jit
def _dot_backward(X,Win,Wout,GO,GX,Parts,N:tl.constexpr,BLOCK:tl.constexpr):
    tile=tl.program_id(0);row=tile*BLOCK+tl.arange(0,BLOCK);k=tl.arange(0,16)
    x,wi,z,a=_dot_hidden(X,Win,row,N)
    go=tl.load(GO+row[:,None]*4+k[None,:],(row[:,None]<N)&(k[None,:]<4),0).to(tl.float32)
    wo=tl.load(Wout+k[:,None]*16+k[None,:],k[:,None]<4,0)
    dz=tl.where(z>0.,tl.dot(go,wo,input_precision='tf32x3'),0.)
    dx=tl.dot(dz,wi,input_precision='tf32x3')
    dwi=tl.dot(tl.trans(dz),x,input_precision='tf32x3')
    dwo=tl.dot(tl.trans(go),a,input_precision='tf32x3')
    tl.store(GX+row[:,None]*4+k[None,:],dx,(row[:,None]<N)&(k[None,:]<4))
    tl.store(Parts+tile*128+k[:,None]*4+k[None,:],dwi,k[None,:]<4)
    tl.store(Parts+tile*128+64+k[:,None]*16+k[None,:],dwo,k[:,None]<4)


class _Optimized(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y,post,win,wout,implementation):
        if implementation not in IMPLEMENTATIONS[1:]:raise ValueError(implementation)
        if x.dtype!=torch.bfloat16 or y.dtype!=x.dtype:raise ValueError('BF16 activations required')
        if any(t.dtype!=torch.float32 for t in (post,win,wout)):raise ValueError('FP32 routing required')
        if not all(t.is_contiguous() for t in (x,y,post,win,wout)):raise ValueError('Contiguous tensors required')
        if x.shape[-1]!=4 or y.shape!=x.shape[:-1] or post.shape!=(*x.shape[:-2],4):raise ValueError('Invalid stream shapes')
        if win.shape!=(16,4) or wout.shape!=(4,16):raise ValueError('Expected4->16->4')
        n=x.numel()//4;out=torch.empty_like(x)
        block=256
        _row_forward[(tr.cdiv(n,block),)](x,y,post,win,wout,out,n,x.shape[-2],block,num_warps=4)
        ctx.save_for_backward(x,y,post,win,wout);ctx.implementation=implementation
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,go):
        x,y,post,win,wout=ctx.saved_tensors;go=go.contiguous()
        n=x.numel()//4;c=x.shape[-2]
        gx=torch.empty_like(x);gy=torch.empty_like(y);gp=torch.empty_like(post)
        dot=ctx.implementation!='rowwise';block=128 if dot else 256
        parts=torch.empty((tr.cdiv(n,block),128),device=x.device,dtype=torch.float32)
        fn=_dot_backward if dot else _row_backward
        fn[(tr.cdiv(n,block),)](x,win,wout,go,gx,parts,n,block,num_warps=4)
        _backward_branch[(n//c,)](y,post,go,gy,gp,n//c,c,tr.next_power_of_2(c),num_warps=4)
        reduced=parts.sum(0)
        return gx,gy,gp,reduced[:64].reshape_as(win).contiguous(),reduced[64:].reshape_as(wout).contiguous(),None


def optimized_stream_mlp(x,y,post,win,wout,implementation):
    return _Optimized.apply(x,y,post,win,wout,implementation)
