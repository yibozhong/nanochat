"""HC scale-all with independent, input-dependent elementwise gates."""
import torch
import triton as tr
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from nanochat.nvidia_connections import NvidiaHyperConnection, NvidiaHyperHead, check_precision
from nanochat.nvidia_ops import nvidia_ops
from nanochat.dynamic_hc_kernels import dynamic_hc_fwd, dynamic_hc_bwd
from nanochat.scaled_reduction import reduce_scale_partials

EPS = 1e-5


def check_inputs(x, p, params):
    if x.ndim != 4 or x.shape[-1] != 4 or not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError('Dynamic HC expects CUDA BF16 [B,T,C,4]')
    c=x.shape[-2];m=x.numel()//(4*c)
    if not 1 <= c <= 1024 or m < 1 or p.shape != (*x.shape[:2],4):
        raise ValueError('Dynamic HC supports nonempty rows with at most1024 channels')
    if any(v.shape != (c,4) for v in params):raise ValueError('Gate parameters must be [C,4]')
    if any(v.dtype != torch.float32 for v in (p,*params)):raise ValueError('Routes/gates must be FP32')
    if not all(v.is_contiguous() and v.device==x.device for v in (x,p,*params)):
        raise ValueError('Dynamic HC requires contiguous colocated inputs')
    return m,c


class _DynamicRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,p,a,b):
        m,c=check_inputs(x,p,(a,b));out=torch.empty(x.shape[:-1],device=x.device,dtype=x.dtype)
        dynamic_hc_fwd[lambda meta:(tr.cdiv(m,meta['BM']),)](
            x,x,p,p,a,b,a,b,out,m,c,False,EPS,tr.next_power_of_2(c))
        ctx.save_for_backward(x,p,a,b)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,go):
        x,p,a,b=ctx.saved_tensors;go=go.contiguous();c=x.shape[-2];m=x.numel()//(4*c)
        gx=torch.empty_like(x);gp=torch.empty_like(p)
        parts=torch.empty((2,tr.cdiv(m,4),c,4),device=x.device,dtype=torch.float32)
        dynamic_hc_bwd[lambda meta:(tr.cdiv(m,meta['BM']),)](
            x,x,p,p,a,b,a,b,go,gx,gx,gp,gp,parts,m,c,False,EPS,tr.next_power_of_2(c))
        ab=reduce_scale_partials(parts)
        return gx,gp,ab[0],ab[1]


class _DynamicWrite(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y,p,h,a,b,pa,pb):
        m,c=check_inputs(x,p,(a,b,pa,pb))
        if y.shape!=x.shape[:-1] or y.dtype!=x.dtype or h.shape!=(*x.shape[:2],4,4) or h.dtype!=torch.float32:
            raise ValueError('Invalid branch/routing shapes or dtypes')
        if not all(v.is_contiguous() and v.device==x.device for v in (y,h)):raise ValueError('Noncontiguous branch/routes')
        out=torch.empty_like(x)
        dynamic_hc_fwd[lambda meta:(tr.cdiv(m,meta['BM']),)](
            x,y,p,h,a,b,pa,pb,out,m,c,True,EPS,tr.next_power_of_2(c))
        ctx.save_for_backward(x,y,p,h,a,b,pa,pb)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,go):
        x,y,p,h,a,b,pa,pb=ctx.saved_tensors;go=go.contiguous();c=x.shape[-2];m=x.numel()//(4*c)
        gx=torch.empty_like(x);gy=torch.empty_like(y);gp=torch.empty_like(p);gh=torch.empty_like(h)
        parts=torch.empty((4,tr.cdiv(m,4),c,4),device=x.device,dtype=torch.float32)
        dynamic_hc_bwd[lambda meta:(tr.cdiv(m,meta['BM']),)](
            x,y,p,h,a,b,pa,pb,go,gx,gy,gp,gh,parts,m,c,True,EPS,tr.next_power_of_2(c))
        ab=reduce_scale_partials(parts)
        return gx,gy,gp,gh,ab[0],ab[1],ab[2],ab[3]


def dynamic_read(x,p,a,b):return _DynamicRead.apply(x,p,a,b)
def dynamic_write(x,y,p,h,a,b,pa,pb):return _DynamicWrite.apply(x,y,p,h,a,b,pa,pb)


def reference_gate(x,a,b):
    u=x*torch.rsqrt(x.square().mean(-2,keepdim=True)+EPS)
    return 2*torch.sigmoid(u*a+b)


def reference_read(x,p,a,b):
    return (x*reference_gate(x,a,b)*p.unsqueeze(-2)).sum(-1)


def reference_write(x,y,p,h,a,b,pa,pb):
    yg=y.unsqueeze(-1)*reference_gate(y.unsqueeze(-1),pa,pb)
    return torch.einsum('btci,btij->btcj',x*reference_gate(x,a,b),h)+yg*p.unsqueeze(-2)


class DynamicAllHyperConnection(NvidiaHyperConnection):
    def __init__(self,width,streams,branch_index,iterations=20,*,use_tf32=True,fused_grad_acc=False,**unused):
        if fused_grad_acc:raise ValueError('Dynamic HC owns ordinary gradients')
        super().__init__(width,streams,branch_index,iterations,use_tf32=use_tf32,fused_grad_acc=False,sinkhorn=False)
        for position in ('read','mix','post'):
            for kind in ('a','b'):setattr(self,position+'_'+kind,nn.Parameter(torch.empty(width,4)))

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        for position in ('read','mix','post'):
            for kind in ('a','b'):getattr(self,position+'_'+kind).zero_()

    def read(self,x):
        check_precision(self.use_tf32);x=x.contiguous();b,t,d,n=x.shape;ops=nvidia_ops()
        raw,ms=ops.mhc_fused_projection(x.view(b*t,d*n),self.phi,norm_weight=None,
            use_tf32=self.use_tf32,fused_grad_x_acc_buffer=None,use_split_k=False)
        pre,post,h=ops.mhc_fused_scale(raw,self.scales,self.bias.unsqueeze(0),ms,n)
        return dynamic_read(x,pre.view(b,t,n).contiguous(),self.read_a,self.read_b),x,(post.view(b,t,n),h.view(b,t,n,n),None)

    def write(self,y,x,mixes):
        p,h,buffer=mixes;assert buffer is None
        return dynamic_write(x,y.contiguous(),p.contiguous(),h.contiguous(),self.mix_a,self.mix_b,self.post_a,self.post_b)


class DynamicHyperHead(NvidiaHyperHead):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gate_a=nn.Parameter(torch.empty(self.width,4));self.gate_b=nn.Parameter(torch.empty(self.width,4))

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters();self.gate_a.zero_();self.gate_b.zero_()

    def forward(self,x):
        check_precision(self.use_tf32);ops=nvidia_ops();b,t,d,n=x.shape
        raw,ms=ops.mhc_fused_projection(x.reshape(b*t,d*n),F.pad(self.phi,(0,0,0,24-n)),
            use_tf32=self.use_tf32,norm_weight=None,fused_grad_x_acc_buffer=None,use_split_k=False)
        pre,_,_=ops.mhc_fused_scale(raw,F.pad(self.scale,(0,2)),F.pad(self.bias,(0,24-n)).unsqueeze(0),ms,n)
        return dynamic_read(x,pre.reshape(b,t,n).contiguous(),self.gate_a,self.gate_b)
