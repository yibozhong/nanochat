"""Optional rowwise channel mixing; same parameters, formula and FP32 policy.

Only one input-stream row of the 4x4 coefficient matrix is live at a time.
Fused variants also compute branch gradients and deterministic dpost partials.
No saved coefficient bank, atomics, BF16 coefficient rounding or new model.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from nanochat.channel_mixing import _backward_branch

# forward BT/BC/warps, backward BT/BC/warps, fuse branch backward.
GEOMETRIES = {
    'split': (32,16,4,128,8,8,False),
    'fused': (32,16,4,64,32,8,True),
    'fused_wide': (32,32,8,128,16,8,True),
}


@tr.jit
def _split4(v):
    a,b=tl.split(tl.reshape(v,(v.shape[0],v.shape[1],2,2)))
    x0,x2=tl.split(a);x1,x3=tl.split(b)
    return x0,x1,x2,x3


@tr.jit
def _select(a,b,c,d,i:tl.constexpr):
    if i==0:return a
    elif i==1:return b
    elif i==2:return c
    else:return d


@tr.jit
def _load_state(X,t,c,N:tl.constexpr,C:tl.constexpr):
    r=tl.arange(0,4)
    x=tl.load(X+(t[:,None,None]*C+c[None,:,None])*4+r[None,None,:],
              (t[:,None,None]<N)&(c[None,:,None]<C),0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,2)*.25+1.1920928955078125e-7)
    x0,x1,x2,x3=_split4(x)
    return x0,x1,x2,x3,inv


@tr.jit
def _row(W,c,v0,v1,v2,v3,i:tl.constexpr,C:tl.constexpr):
    j=tl.arange(0,4)
    off=c[:,None]*64+i*16+j[None,:]*4
    w0=tl.load(W+off,c[:,None]<C,0)
    w1=tl.load(W+off+1,c[:,None]<C,0)
    w2=tl.load(W+off+2,c[:,None]<C,0)
    w3=tl.load(W+off+3,c[:,None]<C,0)
    raw=v0[:,:,None]*w0[None,:,:]
    raw=raw+v1[:,:,None]*w1[None,:,:]
    raw=raw+v2[:,:,None]*w2[None,:,:]
    raw=raw+v3[:,:,None]*w3[None,:,:]
    return raw,w0,w1,w2,w3


@tr.jit
def _channel_row_forward(X,Y,P,W,B,A,O,N:tl.constexpr,C:tl.constexpr,
                         BT:tl.constexpr,BC:tl.constexpr):
    t=tl.program_id(0)*BT+tl.arange(0,BT)
    c=tl.program_id(1)*BC+tl.arange(0,BC);j=tl.arange(0,4)
    x0,x1,x2,x3,inv=_load_state(X,t,c,N,C)
    v0=x0*inv;v1=x1*inv;v2=x2*inv;v3=x3*inv;a=tl.load(A)
    lo=tl.full((BT,BC,4),0.,tl.float32);hi=tl.full((BT,BC,4),0.,tl.float32)
    for i in tl.static_range(4):
        raw,w0,w1,w2,w3=_row(W,c,v0,v1,v2,v3,i,C)
        b=tl.load(B+c[:,None]*16+i*4+j[None,:],c[:,None]<C,0)
        m=b[None,:,:]+a*raw
        xi=_select(x0,x1,x2,x3,i)
        if i<2:lo=lo+xi[:,:,None]*m
        else:hi=hi+xi[:,:,None]*m
    y=tl.load(Y+t[:,None]*C+c[None,:],(t[:,None]<N)&(c[None,:]<C),0).to(tl.float32)
    p=tl.load(P+t[:,None]*4+j[None,:],t[:,None]<N,0)
    out=(lo+hi)+y[:,:,None]*p[:,None,:]
    tl.store(O+(t[:,None,None]*C+c[None,:,None])*4+j[None,None,:],out,
             (t[:,None,None]<N)&(c[None,:,None]<C))


@tr.jit
def _channel_row_backward(X,Y,P,W,B,A,GO,GX,GY,Parts,PostParts,
                         N:tl.constexpr,C:tl.constexpr,BT:tl.constexpr,BC:tl.constexpr,FUSE:tl.constexpr):
    split=tl.program_id(0);cb=tl.program_id(1)
    t=split*BT+tl.arange(0,BT);c=cb*BC+tl.arange(0,BC);j=tl.arange(0,4)
    x0,x1,x2,x3,inv=_load_state(X,t,c,N,C)
    v0=x0*inv;v1=x1*inv;v2=x2*inv;v3=x3*inv;a=tl.load(A)
    go=tl.load(GO+(t[:,None,None]*C+c[None,:,None])*4+j[None,None,:],
               (t[:,None,None]<N)&(c[None,:,None]<C),0).to(tl.float32)
    dv0=tl.full((BT,BC),0.,tl.float32);dv1=tl.full((BT,BC),0.,tl.float32)
    dv2=tl.full((BT,BC),0.,tl.float32);dv3=tl.full((BT,BC),0.,tl.float32)
    dx0=tl.full((BT,BC),0.,tl.float32);dx1=tl.full((BT,BC),0.,tl.float32)
    dx2=tl.full((BT,BC),0.,tl.float32);dx3=tl.full((BT,BC),0.,tl.float32)
    da=tl.full((BT,BC),0.,tl.float32)
    for i in tl.static_range(4):
        raw,w0,w1,w2,w3=_row(W,c,v0,v1,v2,v3,i,C)
        b=tl.load(B+c[:,None]*16+i*4+j[None,:],c[:,None]<C,0)
        m=b[None,:,:]+a*raw
        xi=_select(x0,x1,x2,x3,i);dm=xi[:,:,None]*go
        direct=tl.sum(m*go,2)
        if i==0:dx0=direct
        elif i==1:dx1=direct
        elif i==2:dx2=direct
        else:dx3=direct
        dv0=dv0+tl.sum(dm*w0[None,:,:],2)
        dv1=dv1+tl.sum(dm*w1[None,:,:],2)
        dv2=dv2+tl.sum(dm*w2[None,:,:],2)
        dv3=dv3+tl.sum(dm*w3[None,:,:],2)
        da=da+tl.sum(dm*raw,2)
        for k in tl.static_range(4):
            vk=_select(v0,v1,v2,v3,k)
            dw=tl.sum(dm*(a*vk[:,:,None]),0)
            tl.store(Parts+(split*C+c[:,None])*81+i*16+j[None,:]*4+k,dw,c[:,None]<C)
        tl.store(Parts+(split*C+c[:,None])*81+64+i*4+j[None,:],tl.sum(dm,0),c[:,None]<C)
    dv0=dv0*a;dv1=dv1*a;dv2=dv2*a;dv3=dv3*a
    dot=((dv0*v0+dv1*v1)+(dv2*v2+dv3*v3))*.25
    for k in tl.static_range(4):
        dx=_select(dx0,dx1,dx2,dx3,k)
        dv=_select(dv0,dv1,dv2,dv3,k);vk=_select(v0,v1,v2,v3,k)
        dx=dx+inv*(dv-vk*dot)
        tl.store(GX+(t[:,None]*C+c[None,:])*4+k,dx,(t[:,None]<N)&(c[None,:]<C))
    tl.store(Parts+(split*C+c)*81+80,tl.sum(da,0),c<C)
    if FUSE:
        p=tl.load(P+t[:,None]*4+j[None,:],t[:,None]<N,0)
        y=tl.load(Y+t[:,None]*C+c[None,:],(t[:,None]<N)&(c[None,:]<C),0).to(tl.float32)
        tl.store(GY+t[:,None]*C+c[None,:],tl.sum(go*p[:,None,:],2),(t[:,None]<N)&(c[None,:]<C))
        gp=tl.sum(y[:,:,None]*go,1)
        tl.store(PostParts+(t[:,None]*tl.cdiv(C,BC)+cb)*4+j[None,:],gp,t[:,None]<N)


class _ChannelMixFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y,post,weight,bias,scale,kernel):
        c=x.shape[-2];n=x.numel()//(4*c)
        if not x.is_cuda or x.ndim!=4 or x.shape[-1]!=4 or x.dtype!=torch.bfloat16 or y.dtype!=x.dtype:
            raise ValueError('Expected CUDA BF16 four-stream activations')
        if y.shape!=x.shape[:-1] or post.shape!=(*x.shape[:2],4) or weight.shape!=(c,4,4,4) or bias.shape!=(c,4,4) or scale.shape!=(1,):
            raise ValueError('Invalid channel mixing shapes')
        if not all(p.is_contiguous() and p.device==x.device for p in (x,y,post,weight,bias,scale)):
            raise ValueError('Expected contiguous colocated tensors')
        if any(p.dtype!=torch.float32 for p in (post,weight,bias,scale)):
            raise ValueError('Channel coefficients and parameters remain FP32')
        ft,fc,fw,bt,bc,bw,fuse=GEOMETRIES[kernel]
        out=torch.empty_like(x)
        _channel_row_forward[(tr.cdiv(n,ft),tr.cdiv(c,fc))](x,y,post,weight,bias,scale,out,n,c,ft,fc,num_warps=fw)
        ctx.save_for_backward(x,y,post,weight,bias,scale);ctx.geometry=(bt,bc,bw,fuse)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,go):
        x,y,post,weight,bias,scale=ctx.saved_tensors;go=go.contiguous()
        c=x.shape[-2];n=x.numel()//(4*c);bt,bc,bw,fuse=ctx.geometry
        gx=torch.empty_like(x);gy=torch.empty_like(y)
        parts=torch.empty((tr.cdiv(n,bt),c,81),device=x.device,dtype=torch.float32)
        pp=torch.empty((n,tr.cdiv(c,bc),4),device=x.device,dtype=torch.float32) if fuse else torch.empty_like(post)
        _channel_row_backward[(tr.cdiv(n,bt),tr.cdiv(c,bc))](x,y,post,weight,bias,scale,go,gx,gy,parts,pp,n,c,bt,bc,fuse,num_warps=bw)
        if fuse:gp=pp.sum(1).reshape_as(post)
        else:
            gp=pp
            _backward_branch[(n,)](y,post,go,gy,gp,n,c,tr.next_power_of_2(c))
        reduced=parts.sum(0)
        return gx,gy,gp,reduced[:,:64].reshape_as(weight).contiguous(),reduced[:,64:80].reshape_as(bias).contiguous(),reduced[:,80].sum().reshape_as(scale),None


def channel_mix_fast(x,y,post,weight,bias,scale,*,kernel='fused'):
    return _ChannelMixFast.apply(x,y,post,weight,bias,scale,kernel)
