"""Reader/source/channel static gates on Block AttnRes values only."""
import torch
from torch import nn
from nanochat.attention_residuals import AttnResModel
from nanochat.attnres_scale_kernels import scaled_attn_res_fwd_triton, scaled_attn_res_bwd_triton


class _ScaledAttnRes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prefix, history, nw, pw, scale, eps):
        prefix = prefix if prefix.stride(-1)==1 else prefix.contiguous()
        history = history if history.stride(-1)==1 else history.contiguous()
        out,stats=scaled_attn_res_fwd_triton(prefix,history,nw,pw,scale,eps)
        ctx.save_for_backward(prefix,history,nw,pw,scale,stats)
        return out

    @staticmethod
    def backward(ctx, grad):
        # Unpack once: compatible with non-reentrant activation checkpointing.
        prefix,history,nw,pw,scale,stats=ctx.saved_tensors
        grad=grad if grad.stride(-1)==1 else grad.contiguous()
        dp,dh,dsw,da=scaled_attn_res_bwd_triton(prefix,history,nw,pw,scale,grad,stats)
        return dp,dh,(dsw*pw.float()).to(nw.dtype),(dsw*nw.float()).to(pw.dtype),da,None


def reference_scaled_mix(prefix,history,nw,pw,scale,eps=1e-5):
    values=torch.cat((history,prefix.unsqueeze(1)),dim=1).float()
    keys=values*torch.rsqrt(values.square().mean(-1,keepdim=True)+eps)
    p=(keys*(nw.float()*pw.float())).sum(-1).softmax(-1)
    return (p.unsqueeze(-1)*(values*scale)).sum(1).to(prefix.dtype)


class ScaledAttnResMix(nn.Module):
    def __init__(self, original, entries):
        super().__init__()
        self.norm_weight=original.norm_weight
        self.proj_weight=original.proj_weight
        self.eps=original.eps
        self.value_scale=nn.Parameter(torch.ones(entries,self.norm_weight.numel(),
                                                device=self.norm_weight.device))

    def forward(self,prefix,history):
        shape=prefix.shape
        return _ScaledAttnRes.apply(prefix.reshape(-1,shape[-1]),history,
            self.norm_weight,self.proj_weight,self.value_scale,self.eps).view(shape)


class ScaledAttnResModel(AttnResModel):
    def __init__(self,config):
        # Original backbone and queries drawn first: all-one gates consume no RNG.
        super().__init__(config)
        history=0
        for i,pair in enumerate(self.depth_reads):
            if history:pair[0]=ScaledAttnResMix(pair[0],history+1)
            if i%config.attnres_block_layers==0:history+=1
            pair[1]=ScaledAttnResMix(pair[1],history+1)
        self.output_attnres=ScaledAttnResMix(self.output_attnres,history+1)
