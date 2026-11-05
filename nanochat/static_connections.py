"""Static full-width residual mixing with the original NVIDIA mHC read/write.

Flattening follows persistent [channel, stream] order. Each sublayer has
independent, token-shared weights. There is no extra identity bypass, mixing
normalization, gate, generated matrix, or Sinkhorn. Ordinary BF16 GEMMs and
autograd implement the linear/MLP mixer; NVIDIA kernels implement routing.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from nanochat.nvidia_connections import interleave_phi, check_precision
from nanochat.nvidia_ops import nvidia_ops


class StaticHyperConnection(nn.Module):
    def __init__(self, width, streams, branch_index, iterations=20, *,
                 mixing='linear', use_tf32=True, fused_grad_acc=False, **unused):
        super().__init__()
        if streams != 4 or fused_grad_acc or mixing not in ('linear', 'mlp'):
            raise ValueError('Static mixing requires four streams, linear/mlp, and unfused gradient accumulation')
        self.width, self.streams, self.mixing = width, streams, mixing
        self.use_tf32, self.sinkhorn = use_tf32, False
        self.phi = nn.Parameter(torch.empty(8,4*width))
        self.bias = nn.Parameter(torch.empty(8))
        self.scales = nn.Parameter(torch.empty(2))
        if mixing == 'linear':
            self.mix_weight = nn.Parameter(torch.empty(4*width,4*width))
        else:
            self.mix_down = nn.Parameter(torch.empty(width,4*width))
            self.mix_up = nn.Parameter(torch.empty(4*width,width))
        self.register_buffer('write_identity',torch.eye(4),persistent=False)
        self.compiled_mix = None

    @torch.no_grad()
    def reset_parameters(self):
        # Identical RNG consumption and read/write initialization to mHC.
        original=self.phi.new_empty(24,4*self.width)
        nn.init.normal_(original,std=1/math.sqrt(4*self.width))
        self.phi.copy_(interleave_phi(original)[:8])
        self.bias[:4].fill_(-math.log(3));self.bias[4:].zero_();self.scales.fill_(.01)
        if self.mixing == 'linear':
            nn.init.eye_(self.mix_weight)
        else:
            devices=[self.phi.device.index] if self.phi.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                # ReLU halves the second moment; both matrices are nonzero.
                nn.init.normal_(self.mix_down,std=math.sqrt(2/(4*self.width)))
                nn.init.normal_(self.mix_up,std=math.sqrt(1/self.width))

    def read(self, x):
        check_precision(self.use_tf32)
        x=x.contiguous();b,t,c,n=x.shape;ops=nvidia_ops()
        raw,ms=ops.mhc_fused_projection(x.reshape(b*t,c*n),F.pad(self.phi,(0,0,0,16)),
            norm_weight=None,use_tf32=self.use_tf32,fused_grad_x_acc_buffer=None,use_split_k=False)
        pre,post,_=ops.mhc_fused_scale(raw,F.pad(self.scales,(0,1)),
            F.pad(self.bias,(0,16)).unsqueeze(0),ms,n)
        branch=ops.mhc_fused_aggregate(x,pre.reshape(b,t,n),n,
            use_tf32=self.use_tf32,fused_grad_x_acc_buffer=None)
        return branch,x,post.reshape(b,t,n).contiguous()

    def mix(self, x):
        flat=x.flatten(-2)
        if self.mixing == 'linear':
            y=F.linear(flat,self.mix_weight.to(flat.dtype))
        else:
            y=F.linear(flat,self.mix_down.to(flat.dtype)).relu()
            y=F.linear(y,self.mix_up.to(flat.dtype))
        return y.reshape_as(x)

    def write(self, output, residual, post):
        mixed=(self.compiled_mix or self.mix)(residual)
        b,t,_,_=residual.shape
        # Identity here passes the ALREADY MIXED state into the unchanged TE
        # writer. It does not add or bypass the original residual state.
        identity=self.write_identity.expand(b,t,4,4).contiguous()
        return nvidia_ops().mhc_fused_expand_combine(output.contiguous(),None,
            post,mixed.contiguous(),identity,4,use_tf32=self.use_tf32,
            fused_grad_x_acc_buffer=None)
