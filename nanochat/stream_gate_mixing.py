"""Independent D->D sigmoid gates: BF16 batched GEMMs and fused HC write.

RMS reductions and sigmoid use FP32. Only projection logits and inverse RMS
are saved besides the HC inputs; normalized operands are recomputed backward.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable

RMS_EPS = 1e-6


@tr.jit
def _norm_pack(X,U,R,N:tl.constexpr,C:tl.constexpr,K:tl.constexpr,EPS:tl.constexpr,RECOMPUTE:tl.constexpr):
    t=tl.program_id(0);c=tl.arange(0,K);s=tl.arange(0,4)
    x=tl.load(X+(t*C+c[:,None])*4+s[None,:],c[:,None]<C,0).to(tl.float32)
    if RECOMPUTE:
        r=tl.load(R+t*4+s)
    else:
        r=tl.rsqrt(tl.sum(x*x,0)/C+EPS)
        tl.store(R+t*4+s,r)
    tl.store(U+(s[None,:]*N+t)*C+c[:,None],x*r[None,:],c[:,None]<C)


@tr.jit
def _mix_forward(X,Y,P,H,Z,O,N:tl.constexpr,C:tl.constexpr,BT:tl.constexpr,BC:tl.constexpr):
    t=tl.program_id(0)*BT+tl.arange(0,BT);c=tl.program_id(1)*BC+tl.arange(0,BC)
    s=tl.arange(0,4);mask=(t[:,None]<N)&(c[None,:]<C)
    out=tl.full((BT,BC,4),0.,tl.float32)
    for i in tl.static_range(4):
        x=tl.load(X+(t[:,None]*C+c[None,:])*4+i,mask,0).to(tl.float32)
        z=tl.load(Z+(i*N+t[:,None])*C+c[None,:],mask,0).to(tl.float32)
        h=tl.load(H+t[:,None]*16+i*4+s[None,:],t[:,None]<N,0)
        out+=(x*tl.sigmoid(z))[:,:,None]*h[:,None,:]
    y=tl.load(Y+t[:,None]*C+c[None,:],mask,0).to(tl.float32)
    post=tl.load(P+t[:,None]*4+s[None,:],t[:,None]<N,0)
    out+=y[:,:,None]*post[:,None,:]
    tl.store(O+(t[:,None,None]*C+c[None,:,None])*4+s[None,None,:],out,mask[:,:,None])


@tr.jit
def _mix_backward(X,Y,P,H,Z,GO,DIRECT,GZ,GY,GP,GH,N:tl.constexpr,C:tl.constexpr,K:tl.constexpr):
    t=tl.program_id(0);c=tl.arange(0,K);s=tl.arange(0,4)
    go=tl.load(GO+(t*C+c[:,None])*4+s[None,:],c[:,None]<C,0).to(tl.float32)
    post=tl.load(P+t*4+s);y=tl.load(Y+t*C+c,c<C,0).to(tl.float32)
    tl.store(GY+t*C+c,tl.sum(go*post[None,:],1),c<C)
    tl.store(GP+t*4+s,tl.sum(go*y[:,None],0))
    for i in tl.static_range(4):
        x=tl.load(X+(t*C+c)*4+i,c<C,0).to(tl.float32)
        z=tl.load(Z+(i*N+t)*C+c,c<C,0).to(tl.float32)
        gate=tl.sigmoid(z);h=tl.load(H+t*16+i*4+s);du=tl.sum(go*h[None,:],1)
        tl.store(DIRECT+(t*C+c)*4+i,du*gate,c<C)
        tl.store(GZ+(i*N+t)*C+c,du*x*gate*(1.-gate),c<C)
        tl.store(GH+t*16+i*4+s,tl.sum(go*(x*gate)[:,None],0))


@tr.jit
def _norm_backward(X,R,GU,DIRECT,GX,N:tl.constexpr,C:tl.constexpr,K:tl.constexpr):
    t=tl.program_id(0);c=tl.arange(0,K);s=tl.arange(0,4)
    x=tl.load(X+(t*C+c[:,None])*4+s[None,:],c[:,None]<C,0).to(tl.float32)
    r=tl.load(R+t*4+s)
    gu=tl.load(GU+(s[None,:]*N+t)*C+c[:,None],c[:,None]<C,0).to(tl.float32)
    direct=tl.load(DIRECT+(t*C+c[:,None])*4+s[None,:],c[:,None]<C,0)
    dot=tl.sum(x*gu,0)/C
    gx=direct+r[None,:]*(gu-x*(r*r*dot)[None,:])
    tl.store(GX+(t*C+c[:,None])*4+s[None,:],gx,c[:,None]<C)


def project_gates(x,weight):
    """Stream-major BF16 logits and token/stream FP32 inverse RMS."""
    c=x.shape[-2];n=x.numel()//(4*c)
    u=torch.empty((4,n,c),device=x.device,dtype=x.dtype)
    r=torch.empty((n,4),device=x.device,dtype=torch.float32)
    _norm_pack[(n,)](x,u,r,n,c,tr.next_power_of_2(c),RMS_EPS,False,num_warps=8)
    return torch.bmm(u,weight.to(x.dtype).transpose(1,2)),r


class _StreamGateMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y,post,h,weight):
        if x.dtype!=torch.bfloat16 or y.dtype!=x.dtype or not x.is_cuda:
            raise ValueError('Stream gates require CUDA BF16 activations')
        if not all(t.is_contiguous() for t in (x,y,post,h,weight)):
            raise ValueError('Stream gate inputs must be contiguous')
        c=x.shape[-2];n=x.numel()//(4*c)
        if x.shape[-1]!=4 or weight.shape!=(4,c,c) or y.shape!=x.shape[:-1]:
            raise ValueError('Expected four streams, [4,D,D] weights, and matching branch')
        if post.shape!=(*x.shape[:-2],4) or h.shape!=(*x.shape[:-2],4,4):
            raise ValueError('Invalid HC routing shapes')
        if any(t.dtype!=torch.float32 for t in (post,h,weight)):
            raise ValueError('Gate master weights and HC coefficients must be FP32')
        z,r=project_gates(x,weight);out=torch.empty_like(x)
        _mix_forward[(tr.cdiv(n,32),tr.cdiv(c,32))](x,y,post,h,z,out,n,c,32,32,num_warps=4)
        ctx.save_for_backward(x,y,post,h,weight,z,r)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,go):
        x,y,post,h,weight,z,r=ctx.saved_tensors
        go=go.contiguous();c=x.shape[-2];n=x.numel()//(4*c)
        direct=torch.empty_like(x,dtype=torch.float32)
        gz=torch.empty_like(z);gy=torch.empty_like(y);gp=torch.empty_like(post);gh=torch.empty_like(h)
        _mix_backward[(n,)](x,y,post,h,z,go,direct,gz,gy,gp,gh,n,c,tr.next_power_of_2(c),num_warps=8)
        u=torch.empty_like(gz)
        _norm_pack[(n,)](x,u,r,n,c,tr.next_power_of_2(c),RMS_EPS,True,num_warps=8)
        # Ordinary BF16 Linear autograd rounds each GEMM result to BF16,
        # then casts the weight gradient back to its FP32 master parameter.
        gw=torch.bmm(gz.transpose(1,2),u).float()
        del u
        gu=torch.bmm(gz,weight.to(x.dtype));gx=torch.empty_like(x)
        _norm_backward[(n,)](x,r,gu,direct,gx,n,c,tr.next_power_of_2(c),num_warps=8)
        return gx,gy,gp,gh,gw


def stream_gate_mix(x,y,post,h,weight):
    return _StreamGateMix.apply(x,y,post,h,weight)


def stream_gate_reference(x,y,post,h,weight):
    """Autograd reference with identical projection precision and formula."""
    c=x.shape[-2];n=x.numel()//(4*c);value=x.float()
    u=(value*torch.rsqrt(value.square().mean(-2,keepdim=True)+RMS_EPS)).to(x.dtype)
    u=u.reshape(n,c,4).permute(2,0,1).contiguous()
    z=torch.bmm(u,weight.to(x.dtype).transpose(1,2))
    g=torch.sigmoid(z.float()).permute(1,2,0).reshape_as(x);gated=value*g
    out=gated[...,0,None]*h[...,0,:].unsqueeze(-2)
    for i in range(1,4):out=out+gated[...,i,None]*h[...,i,:].unsqueeze(-2)
    return (out+y.float().unsqueeze(-1)*post.unsqueeze(-2)).to(x.dtype)


compiled_stream_gate_reference=torch.compile(stream_gate_reference,fullgraph=True,dynamic=False)
