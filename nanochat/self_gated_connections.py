"""Diagonal learned SiLU gate on the residual input of shared no-Sinkhorn HC.

Each scalar gates itself: s(x) = x * SiLU(a*x+b). Parameters a and b have
shape [channel, stream]; no reduction or cross-stream/channel gate dependency.
The pointwise operation is compiled separately; NVIDIA routing/write stay intact.
"""
import torch
from torch import nn
from torch.nn import functional as F

from nanochat.nvidia_connections import NvidiaHyperConnection


SILU_IDENTITY_BIAS = 1.2784645427610737


def self_gate_reference(x, weight, bias):
    value=x.float()
    return (value*F.silu(value*weight.float()+bias.float())).to(x.dtype)


self_gate = torch.compile(self_gate_reference, fullgraph=True, dynamic=False)


class SelfGatedHyperConnection(NvidiaHyperConnection):
    def __init__(self, width, streams, branch_index, iterations=20, *,
                 use_tf32=True, fused_grad_acc=False, **unused):
        if fused_grad_acc:
            raise ValueError('Self-gated HC uses ordinary gradient accumulation')
        super().__init__(width, streams, branch_index, iterations,
                         use_tf32=use_tf32, fused_grad_acc=False, sinkhorn=False)
        self.gate_weight=nn.Parameter(torch.empty(width,streams))
        self.gate_bias=nn.Parameter(torch.empty(width,streams))

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        self.gate_weight.zero_()
        self.gate_bias.fill_(SILU_IDENTITY_BIAS)

    def write(self, output, residual, mixes):
        return super().write(output, self_gate(residual,self.gate_weight,self.gate_bias), mixes)
