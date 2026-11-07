"""Gate the shared HC mixing output, with gates still generated from its input.

output = (x @ H) * sigmoid(RMS_channel(x_s) @ Wg_s.T) + y * post.
The branch write remains outside the gate. No mixed-state activation is saved;
backward recomputes that four-stream mix using FP32 arithmetic.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from nanochat.stream_gate_mixing import project_gates,_norm_pack,_norm_backward,RMS_EPS


@tr.jit
def _post_forward(X,Y,P,H,Z,O,N:tl.constexpr,C:tl.constexpr,BT:tl.constexpr,BC:tl.constexpr):
    t=tl.program_id(0)*BT+tl.arange(0,BT);c=tl.program_id(1)*BC+tl.arange(0,BC)
    s=tl.arange(0,4);mask=(t[:,None]<N)&(c[None,:]<C)
    mixed=tl.full((BT,BC,4),0.,tl.float32)
    for i in tl.static_range(4):
        x=tl.load(X+(t[:,None]*C+c[None,:])*4+i,mask,0).to(tl.float32)
        h=tl.load(H+t[:,None]*16+i*4+s[None,:],t[:,None]<N,0)
        mixed+=x[:,:,None]*h[:,None,:]
    z=tl.load(Z+(s[None,None,:]*N+t[:,None,None])*C+c[None,:,None],mask[:,:,None],0).to(tl.float32)
    y=tl.load(Y+t[:,None]*C+c[None,:],mask,0).to(tl.float32)
    post=tl.load(P+t[:,None]*4+s[None,:],t[:,None]<N,0)
    out=mixed*tl.sigmoid(z)+y[:,:,None]*post[:,None,:]
    tl.store(O+(t[:,None,None]*C+c[None,:,None])*4+s[None,None,:],out,mask[:,:,None])


@tr.jit
def _post_backward(X,Y,P,H,Z,GO,DIRECT,GZ,GY,GP,GH,N:tl.constexpr,C:tl.constexpr,K:tl.constexpr):
    t=tl.program_id(0);c=tl.arange(0,K);s=tl.arange(0,4)
    x=tl.load(X+(t*C+c[:,None])*4+s[None,:],c[:,None]<C,0).to(tl.float32)
    go=tl.load(GO+(t*C+c[:,None])*4+s[None,:],c[:,None]<C,0).to(tl.float32)
    z=tl.load(Z+(s[None,:]*N+t)*C+c[:,None],c[:,None]<C,0).to(tl.float32)
    gate=tl.sigmoid(z)
    mixed=tl.full((K,4),0.,tl.float32)
    for i in tl.static_range(4):
        xi=tl.sum(tl.where(s[None,:]==i,x,0.),1)
        h=tl.load(H+t*16+i*4+s)
        mixed+=xi[:,None]*h[None,:]
    tl.store(GZ+(s[None,:]*N+t)*C+c[:,None],go*mixed*gate*(1.-gate),c[:,None]<C)
    gm=go*gate
    direct=tl.full((K,4),0.,tl.float32)
    for i in tl.static_range(4):
        xi=tl.sum(tl.where(s[None,:]==i,x,0.),1)
        h=tl.load(H+t*16+i*4+s)
        dx=tl.sum(gm*h[None,:],1)
        direct+=tl.where(s[None,:]==i,dx[:,None],0.)
        tl.store(GH+t*16+i*4+s,tl.sum(gm*xi[:,None],0))
    tl.store(DIRECT+(t*C+c[:,None])*4+s[None,:],direct,c[:,None]<C)
    post=tl.load(P+t*4+s);y=tl.load(Y+t*C+c,c<C,0).to(tl.float32)
    tl.store(GY+t*C+c,tl.sum(go*post[None,:],1),c<C)
    tl.store(GP+t*4+s,tl.sum(go*y[:,None],0))


class _PostStreamGateMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y,post,h,weight):
        if x.dtype!=torch.bfloat16 or y.dtype!=x.dtype or not x.is_cuda:
            raise ValueError('Post-mix stream gates require CUDA BF16 activations')
        if not all(t.is_contiguous() for t in (x,y,post,h,weight)):
            raise ValueError('Post-mix stream gate inputs must be contiguous')
        c=x.shape[-2];n=x.numel()//(4*c)
        if x.shape[-1]!=4 or weight.shape!=(4,c,c) or y.shape!=x.shape[:-1]:
            raise ValueError('Expected four streams, [4,D,D] weights, and matching branch')
        if post.shape!=(*x.shape[:-2],4) or h.shape!=(*x.shape[:-2],4,4):
            raise ValueError('Invalid HC routing shapes')
        if any(t.dtype!=torch.float32 for t in (post,h,weight)):
            raise ValueError('Gate master weights and HC coefficients must be FP32')
        z,r=project_gates(x,weight);out=torch.empty_like(x)
        _post_forward[(tr.cdiv(n,32),tr.cdiv(c,32))](x,y,post,h,z,out,n,c,32,32,num_warps=4)
        ctx.save_for_backward(x,y,post,h,weight,z,r)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,go):
        x,y,post,h,weight,z,r=ctx.saved_tensors
        go=go.contiguous();c=x.shape[-2];n=x.numel()//(4*c)
        direct=torch.empty_like(x,dtype=torch.float32)
        gz=torch.empty_like(z);gy=torch.empty_like(y);gp=torch.empty_like(post);gh=torch.empty_like(h)
        _post_backward[(n,)](x,y,post,h,z,go,direct,gz,gy,gp,gh,n,c,tr.next_power_of_2(c),num_warps=8)
        u=torch.empty_like(gz)
        _norm_pack[(n,)](x,u,r,n,c,tr.next_power_of_2(c),RMS_EPS,True,num_warps=8)
        gw=torch.bmm(gz.transpose(1,2),u).float();del u
        gu=torch.bmm(gz,weight.to(x.dtype));gx=torch.empty_like(x)
        _norm_backward[(n,)](x,r,gu,direct,gx,n,c,tr.next_power_of_2(c),num_warps=8)
        return gx,gy,gp,gh,gw


def post_stream_gate_mix(x,y,post,h,weight):
    return _PostStreamGateMix.apply(x,y,post,h,weight)


def post_stream_gate_reference(x,y,post,h,weight):
    c=x.shape[-2];n=x.numel()//(4*c);value=x.float()
    u=(value*torch.rsqrt(value.square().mean(-2,keepdim=True)+RMS_EPS)).to(x.dtype)
    u=u.reshape(n,c,4).permute(2,0,1).contiguous()
    z=torch.bmm(u,weight.to(x.dtype).transpose(1,2))
    gate=torch.sigmoid(z.float()).permute(1,2,0).reshape_as(x)
    mixed=value[...,0,None]*h[...,0,:].unsqueeze(-2)
    for i in range(1,4):mixed=mixed+value[...,i,None]*h[...,i,:].unsqueeze(-2)
    return (mixed*gate+y.float().unsqueeze(-1)*post.unsqueeze(-2)).to(x.dtype)


compiled_post_stream_gate_reference=torch.compile(post_stream_gate_reference,fullgraph=True,dynamic=False)
