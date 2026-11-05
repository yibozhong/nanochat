"""Unchanged NVIDIA H_pre/H_post, local channel-wise unconstrained H_res."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from nanochat.channel_mixing import channel_mix
from nanochat.nvidia_connections import interleave_phi, check_precision
from nanochat.nvidia_ops import nvidia_ops


class ChannelHyperConnection(nn.Module):
    def __init__(self, width, streams, branch_index, iterations=20, *,
                 use_tf32=True, fused_grad_acc=False, **unused):
        super().__init__()
        if streams != 4 or fused_grad_acc:
            raise ValueError('Local mixing uses four streams and ordinary gradient accumulation')
        self.width, self.streams = width, streams
        self.use_tf32 = use_tf32
        self.sinkhorn = False
        self.phi = nn.Parameter(torch.empty(8,4*width))
        self.bias = nn.Parameter(torch.empty(8))
        self.scales = nn.Parameter(torch.empty(2))
        self.mix_weight = nn.Parameter(torch.empty(width,4,4,4))
        self.mix_bias = nn.Parameter(torch.empty(width,4,4))
        self.mix_scale = nn.Parameter(torch.empty(1))

    @torch.no_grad()
    def reset_parameters(self):
        # Consume exactly the old generator's RNG draws. H_pre/H_post and the
        # downstream initialization then match ordinary mHC at the same seed.
        original = self.phi.new_empty(24,4*self.width)
        nn.init.normal_(original, std=1/math.sqrt(4*self.width))
        self.phi.copy_(interleave_phi(original)[:8])
        self.bias[:4].fill_(-math.log(3)); self.bias[4:].zero_()
        self.scales.fill_(.01)
        devices = [self.phi.device.index] if self.phi.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            nn.init.normal_(self.mix_weight, std=.5)
        self.mix_bias.copy_(torch.eye(4,device=self.phi.device).expand(self.width,4,4))
        self.mix_scale.fill_(.01)

    def read(self, x):
        check_precision(self.use_tf32)
        x=x.contiguous(); b,t,c,n=x.shape
        ops=nvidia_ops()
        # TE's small projection/scale ABI expects 24 rows. Padding contains no
        # learned unused parameters, and is independent of the local generator.
        raw,ms=ops.mhc_fused_projection(x.reshape(b*t,c*n),F.pad(self.phi,(0,0,0,16)),
            norm_weight=None,use_tf32=self.use_tf32,
            fused_grad_x_acc_buffer=None,use_split_k=False)
        pre,post,_=ops.mhc_fused_scale(raw,F.pad(self.scales,(0,1)),
            F.pad(self.bias,(0,16)).unsqueeze(0),ms,n)
        branch=ops.mhc_fused_aggregate(x,pre.reshape(b,t,n),n,
            use_tf32=self.use_tf32,fused_grad_x_acc_buffer=None)
        return branch,x,post.reshape(b,t,n).contiguous()

    def write(self, output, residual, post):
        return channel_mix(residual,output.contiguous(),post,
            self.mix_weight,self.mix_bias,self.mix_scale)
