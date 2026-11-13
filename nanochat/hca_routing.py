"""FP32 scalar routing and confidence VJP without expanded FP32 activations."""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable


@tr.jit
def _router_fwd(X,W,B,Y,N:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,BR:tl.constexpr):
    row=tl.program_id(0)*BR+tl.arange(0,BR)
    d=tl.arange(0,BD)
    x=tl.load(X+row[:,None]*D+d[None,:],(row[:,None]<N)&(d[None,:]<D),0).to(tl.float32)
    w=tl.load(W+d,d<D,0)
    y=tl.sum(x*w[None,:],1)+tl.load(B)
    tl.store(Y+row,y,row<N)


@tr.jit
def _router_bwd(X,W,G,DX,DW,DB,N:tl.constexpr,D:tl.constexpr,BM:tl.constexpr,BD:tl.constexpr):
    block,dc=tl.program_id(0),tl.program_id(1)
    rows=block*BM+tl.arange(0,BM);d=dc*BD+tl.arange(0,BD)
    valid=(rows[:,None]<N)&(d[None,:]<D)
    x=tl.load(X+rows[:,None]*D+d[None,:],valid,0).to(tl.float32)
    g=tl.load(G+rows,rows<N,0)
    w=tl.load(W+d,d<D,0)
    tl.store(DX+rows[:,None]*D+d[None,:],g[:,None]*w[None,:],valid)
    tl.store(DW+block*D+d,tl.sum(g[:,None]*x,0),d<D)
    if dc==0:tl.store(DB+block,tl.sum(g,0))


class _Router(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,bias):
        x=x.contiguous();n=x.numel()//x.shape[-1];d=x.shape[-1]
        out=torch.empty(x.shape[:-1],device=x.device,dtype=torch.float32)
        _router_fwd[(tr.cdiv(n,4),)](x,w,bias,out,n,d,tr.next_power_of_2(d),4,
                                   num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(x,w)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        x,w=ctx.saved_tensors;n=x.numel()//x.shape[-1];d=x.shape[-1]
        parts=tr.cdiv(n,128)
        dw=torch.empty((parts,d),device=x.device,dtype=torch.float32)
        db=torch.empty(parts,device=x.device,dtype=torch.float32)
        dx=torch.empty_like(x)
        _router_bwd[(parts,tr.cdiv(d,128))](x,w,grad.contiguous(),dx,dw,db,n,d,128,128,
                                         num_warps=4,enable_fp_fusion=False)
        return dx,dw.sum(0)[None],db.sum().view(1)


def router_linear(x,w,bias):
    if not x.is_cuda or x.dtype not in (torch.float32,torch.bfloat16):
        raise ValueError('Fused routing requires CUDA FP32/BF16 input')
    if w.shape != (1,x.shape[-1]) or bias.shape != (1,) or w.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError('Fused routing requires one FP32 linear output')
    return _Router.apply(x,w,bias)


@tr.jit
def _confidence_vjp(G,Y,O,N:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,BR:tl.constexpr):
    row=tl.program_id(0)*BR+tl.arange(0,BR);d=tl.arange(0,BD)
    off=row[:,None]*D+d[None,:]
    valid=(row[:,None]<N)&(d[None,:]<D)
    g=tl.load(G+off,valid,0).to(tl.float32);y=tl.load(Y+off,valid,0).to(tl.float32)
    tl.store(O+row,tl.sum(g*y,1),row<N)


def confidence_gradient(grad,output):
    n=output.shape[0]*output.shape[1];d=output.shape[2]*output.shape[3]
    result=torch.empty(output.shape[:2],device=output.device,dtype=torch.float32)
    _confidence_vjp[(tr.cdiv(n,4),)](grad.contiguous(),output.contiguous(),result,n,d,
                                   tr.next_power_of_2(d),4,num_warps=4,enable_fp_fusion=False)
    return result
