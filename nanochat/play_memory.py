"""Continuous play memory in a vanilla FFN, with an exact blockwise scan.

Each sequence starts at zero. Equalities retain the old state; this also
defines the subgradient at a clamp boundary. No persistent training state.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def play_reference(u, radius, weight):
    """Independent serial oracle; keeps the input dtype (FP64 for checks)."""
    state = torch.zeros_like(u[:, 0])
    values = []
    for current in u.unbind(1):
        lo, hi = current - radius, current + radius
        state = torch.where(state < lo, lo, torch.where(state > hi, hi, state))
        values.append(F.relu(current + weight * state).square())
    return torch.stack(values, dim=1)


class PlayActivation(nn.Module):
    def __init__(self, width, radius_init=.25):
        super().__init__()
        self.raw_radius = nn.Parameter(torch.full((width,), math.log(math.expm1(radius_init))))
        self.memory_weight = nn.Parameter(torch.zeros(width))

    def forward(self, u):
        from nanochat.play_memory_kernels import play_relu2
        return play_relu2(u, F.softplus(self.raw_radius), self.memory_weight)


class PlayMLP(nn.Module):
    def __init__(self, original, radius_init=.25):
        super().__init__()
        # Reuse the already constructed projections; consume no extra RNG.
        self.c_fc, self.c_proj = original.c_fc, original.c_proj
        self.play = PlayActivation(self.c_fc.out_features, radius_init)

    def forward(self, x):
        return self.c_proj(self.play(self.c_fc(x)))
