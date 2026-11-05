"""Isolated speed-only models: generalized connectors, unchanged backbone.

The pinned NVIDIA projection and other public APIs only support n=4. For
the *matched* n sweep both methods use the same compiled cuBLAS FP32/TF32
projection, parameter-free RMS, fused Triton aggregation/write, and head.
mHC retains NVIDIA's20-step log-Sinkhorn arithmetic. These are generalized
benchmark implementations, not a claim of official NVIDIA8/16/32 support.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
from nanochat.connection_model import ConnectionConfig,ConnectionModel
from nanochat.gpt import norm
from nanochat.nvidia_connections import NvidiaHyperConnection,NvidiaHyperHead
from nanochat.stream_mlp_connections import StreamMLPHyperConnection
from scripts.stream_scaling_ops import Aggregate,Write,MLP,Sinkhorn

# This module is imported only by the isolated benchmark.
torch._dynamo.config.recompile_limit=64


@torch.compile(fullgraph=True)
def _projection_fwd(x,phi):
    xf=x.float()
    return F.linear(xf,phi),xf.square().mean(-1)


@torch.compile(fullgraph=True)
def _projection_bwd(x,phi,gh,gm):
    xf=x.float()
    gx=gh@phi+xf*(2*gm/x.shape[-1]).unsqueeze(-1)
    gp=gh.T@xf
    return gx.to(x.dtype),gp


class Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,phi):
        ctx.save_for_backward(x,phi)
        return _projection_fwd(x,phi)

    @staticmethod
    @once_differentiable
    def backward(ctx,gh,gm):
        x,phi=ctx.saved_tensors
        return _projection_bwd(x,phi,gh.contiguous(),gm.contiguous())


@torch.compile(fullgraph=True)
def scale(raw,ms,alpha,beta,n,mhc):
    r=(ms+torch.finfo(torch.float32).eps).sqrt().reciprocal().unsqueeze(-1)
    pre=torch.sigmoid(raw[:,:n]*alpha[0]*r+beta[:n])
    post=2*torch.sigmoid(raw[:,n:2*n]*alpha[1]*r+beta[n:2*n])
    residual=raw[:,2*n:]*alpha[2]*r+beta[2*n:] if mhc else raw[:,:0]
    return pre.contiguous(),post.contiguous(),residual.contiguous()


@torch.compile(fullgraph=True)
def head_scale(raw,ms,alpha,beta):
    return torch.sigmoid(raw*alpha/(ms[:,None]+torch.finfo(torch.float32).eps).sqrt()+beta)


def hadamard(n,device):
    q=torch.ones((1,1),device=device)
    while q.shape[0]<n:q=torch.cat((torch.cat((q,q),1),torch.cat((q,-q),1)),0)
    return q/math.sqrt(n)


class GeneralConnection(nn.Module):
    def __init__(self,d,n,kind):
        super().__init__();self.d=d;self.n=n;self.kind=kind
        size=2*n+n*n if kind=='mhc' else 2*n
        self.phi=nn.Parameter(torch.empty(size,n*d))
        self.scales=nn.Parameter(torch.full((3 if kind=='mhc' else 2,),.01))
        self.bias=nn.Parameter(torch.zeros(size))
        nn.init.normal_(self.phi,std=1/math.sqrt(n*d))
        with torch.no_grad():
            self.bias[:n].fill_(-math.log(n-1))
            if kind=='mhc':self.bias[2*n:].reshape(n,n).copy_(torch.eye(n))
        if kind=='mlp':
            eye=torch.eye(n);q=hadamard(n,'cpu')
            # Identity initial mixer with m=4n; no runtime residual bypass.
            self.win=nn.Parameter(torch.cat((eye,-eye,q,-q),0))
            self.wout=nn.Parameter(.5*torch.cat((eye,-eye,q.T,-q.T),1))

    def read(self,x):
        b,t,d,n=x.shape;flat=x.reshape(b*t,d,n)
        raw,ms=Projection.apply(flat.reshape(b*t,d*n),self.phi)
        pre,post,res=scale(raw,ms,self.scales,self.bias,n,self.kind=='mhc')
        memory=Sinkhorn.apply(res.reshape(b*t,n,n)) if self.kind=='mhc' else None
        return Aggregate.apply(flat,pre).reshape(b,t,d),x,(post,memory)

    def write(self,y,x,state):
        b,t,d,n=x.shape;post,memory=state
        if self.kind=='mhc':out=Write.apply(x.reshape(b*t,d,n),y.reshape(b*t,d),post,memory)
        else:out=MLP.apply(x.reshape(b*t,d,n),y.reshape(b*t,d),post,self.win,self.wout)
        return out.reshape_as(x)


class GeneralHead(nn.Module):
    def __init__(self,d,n):
        super().__init__();self.n=n
        self.phi=nn.Parameter(torch.randn(n,n*d)/math.sqrt(n*d))
        self.scale=nn.Parameter(torch.full((1,),.01))
        self.bias=nn.Parameter(torch.full((n,),-math.log(n-1)))

    def forward(self,x):
        b,t,d,n=x.shape;raw,ms=Projection.apply(x.reshape(b*t,d*n),self.phi)
        pre=head_scale(raw,ms,self.scale,self.bias)
        return Aggregate.apply(x.reshape(b*t,d,n),pre).reshape(b,t,d)


class ScalingModel(nn.Module):
    def __init__(self,kind,n,*,width=512,depth=8,vocab=32768,anchor=False):
        super().__init__();self.n=n
        torch.manual_seed(42)
        self.base=ConnectionModel(ConnectionConfig(width=width,depth=depth,heads=4 if width==512 else 2,
            vocab_size=vocab,sequence_length=2048)).prepare(compile_blocks=False)
        # Zero-initialized branches are unrepresentative of the trained graph.
        with torch.no_grad():
            for b in self.base.blocks:
                b.attn.c_proj.weight.normal_(std=.02);b.mlp.c_proj.weight.normal_(std=.02)
        self.attention=[torch.compile(b.attention_branch,fullgraph=True) for b in self.base.blocks]
        self.ffn=[torch.compile(b.mlp_branch,fullgraph=True) for b in self.base.blocks]
        modules=[]
        for i in range(2*depth):
            if anchor:
                assert n==4
                module=(NvidiaHyperConnection(width,4,i) if kind=='mhc' else
                        StreamMLPHyperConnection(width,4,i,compact_routing=True,stream_mlp_kernel='rowwise'))
                module.reset_parameters()
            else:module=GeneralConnection(width,n,kind)
            modules.append(module)
        self.connections=nn.ModuleList(modules).cuda()
        self.head=(NvidiaHyperHead(width,4) if anchor else GeneralHead(width,n)).cuda()
        if anchor:self.head.reset_parameters()

    def hidden(self,ids):
        x=norm(self.base.wte(ids)).unsqueeze(-1).expand(*ids.shape,self.base.config.width,self.n).contiguous()
        cos=self.base.cos[:,:ids.shape[1]];sin=self.base.sin[:,:ids.shape[1]]
        for i in range(len(self.base.blocks)):
            c=self.connections[2*i];h,skip,state=c.read(x)
            x=c.write(self.attention[i](h,cos,sin),skip,state)
            c=self.connections[2*i+1];h,skip,state=c.read(x)
            x=c.write(self.ffn[i](h),skip,state)
        return norm(self.head(x))

    def forward(self,ids,targets):
        from cut_cross_entropy import linear_cross_entropy
        h=self.hidden(ids);w=self.base.lm_head.weight.to(h.dtype)
        return linear_cross_entropy(h,w,targets,softcap=15.,shift=0,reduction='mean',filter_eps=None,
            filter_e_grad=False,filter_c_grad=False,accum_e_fp32=True,accum_c_fp32=True)
