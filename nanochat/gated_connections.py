"""Shared no-Sinkhorn HC with input-side channel-local SiLU gates."""
import torch
from torch import nn

from nanochat.gated_mixing import gated_mix, SILU_IDENTITY_BIAS
from nanochat.nvidia_connections import NvidiaHyperConnection


class GatedHyperConnection(NvidiaHyperConnection):
    def __init__(self, width, streams, branch_index, iterations=20, *,
                 use_tf32=True, fused_grad_acc=False, **unused):
        if fused_grad_acc:
            raise ValueError('Gated HC owns ordinary per-call gradients')
        super().__init__(width, streams, branch_index, iterations,
                         use_tf32=use_tf32, fused_grad_acc=False, sinkhorn=False)
        self.gate_weight=nn.Parameter(torch.empty(width,4,4))
        self.gate_bias=nn.Parameter(torch.empty(width,4))

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        self.gate_weight.zero_()
        self.gate_bias.fill_(SILU_IDENTITY_BIAS)

    def write(self, output, residual, mixes):
        post,h,buffer=mixes
        assert buffer is None
        return gated_mix(residual,output.contiguous(),post.contiguous(),h.contiguous(),
                         self.gate_weight,self.gate_bias)
