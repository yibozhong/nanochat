"""Shared dynamic no-Sinkhorn HC with static channel/input-stream scales."""
import torch
from torch import nn

from nanochat.nvidia_connections import NvidiaHyperConnection
from nanochat.scaled_mixing import scaled_mix


class ScaledHyperConnection(NvidiaHyperConnection):
    def __init__(self, width, streams, branch_index, iterations=20, *,
                 use_tf32=True, fused_grad_acc=False, **unused):
        if fused_grad_acc:
            raise ValueError('Scaled HC owns ordinary per-call gradients')
        super().__init__(width,streams,branch_index,iterations,use_tf32=use_tf32,
                         fused_grad_acc=False,sinkhorn=False)
        self.input_scale=nn.Parameter(torch.empty(width,4))

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        self.input_scale.fill_(1.)

    def write(self, output, residual, mixes):
        post,h,buffer=mixes
        assert buffer is None
        return scaled_mix(residual,output.contiguous(),post.contiguous(),h.contiguous(),self.input_scale)
