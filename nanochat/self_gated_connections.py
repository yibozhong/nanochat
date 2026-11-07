"""Elementwise input-times-SiLU on the residual input of shared no-Sinkhorn HC.

Each scalar gates itself: s(x) = x * SiLU(x) = x**2 * sigmoid(x). No gate parameters,
projection, extra normalization, or cross-stream/channel gate dependency.
The pointwise operation is compiled separately; NVIDIA routing/write stay intact.
"""
import torch
from torch.nn import functional as F

from nanochat.nvidia_connections import NvidiaHyperConnection


def self_gate_reference(x):
    value=x.float()
    return (value*F.silu(value)).to(x.dtype)


self_gate = torch.compile(self_gate_reference, fullgraph=True, dynamic=False)


class SelfGatedHyperConnection(NvidiaHyperConnection):
    def __init__(self, width, streams, branch_index, iterations=20, *,
                 use_tf32=True, fused_grad_acc=False, **unused):
        if fused_grad_acc:
            raise ValueError('Self-gated HC uses ordinary gradient accumulation')
        super().__init__(width, streams, branch_index, iterations,
                         use_tf32=use_tf32, fused_grad_acc=False, sinkhorn=False)

    def write(self, output, residual, mixes):
        return super().write(output, self_gate(residual), mixes)
