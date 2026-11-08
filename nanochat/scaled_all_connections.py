"""Static channel/source scales for residual mixing, read, post and final head."""
import torch
from torch import nn
from torch.nn import functional as F
from nanochat.nvidia_connections import NvidiaHyperHead,check_precision
from nanochat.nvidia_ops import nvidia_ops
from nanochat.scaled_connections import ScaledHyperConnection
from nanochat.scaled_mixing import scaled_mix
from nanochat.scaled_aggregate import scaled_aggregate


class AllScaledHyperConnection(ScaledHyperConnection):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.read_scale=nn.Parameter(torch.empty(self.width,4))
        self.post_scale=nn.Parameter(torch.empty(self.width,4))

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        self.read_scale.fill_(1.)
        self.post_scale.fill_(1.)

    def read(self,x):
        check_precision(self.use_tf32)
        x=x.contiguous();b,t,d,n=x.shape;ops=nvidia_ops()
        # Routing still reads the original, unscaled state.
        raw,ms=ops.mhc_fused_projection(x.view(b*t,d*n),self.phi,
            norm_weight=None,use_tf32=self.use_tf32,fused_grad_x_acc_buffer=None,use_split_k=False)
        pre,post,comb=ops.mhc_fused_scale(raw,self.scales,self.bias.unsqueeze(0),ms,n)
        branch=scaled_aggregate(x,pre.view(b,t,n).contiguous(),self.read_scale)
        return branch,x,(post.view(b,t,n),comb.view(b,t,n,n),None)

    def write(self,output,residual,mixes):
        post,h,buffer=mixes
        assert buffer is None
        return scaled_mix(residual,output.contiguous(),post.contiguous(),h.contiguous(),
                          self.input_scale,self.post_scale)


class ScaledHyperHead(NvidiaHyperHead):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.input_scale=nn.Parameter(torch.empty(self.width,4))

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        self.input_scale.fill_(1.)

    def forward(self,x):
        check_precision(self.use_tf32)
        ops=nvidia_ops();b,t,d,n=x.shape
        phi=F.pad(self.phi,(0,0,0,24-n))
        alpha=F.pad(self.scale,(0,2));beta=F.pad(self.bias,(0,24-n)).unsqueeze(0)
        raw,ms=ops.mhc_fused_projection(x.reshape(b*t,d*n),phi,
            use_tf32=self.use_tf32,norm_weight=None,fused_grad_x_acc_buffer=None,use_split_k=False)
        pre,_,_=ops.mhc_fused_scale(raw,alpha,beta,ms,n)
        return scaled_aggregate(x,pre.reshape(b,t,n).contiguous(),self.input_scale)
