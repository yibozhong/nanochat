"""Tokenwise SwiGLU features before the existing D -> vocabulary matrix.

h is the original LM-head input, after the backbone's final RMSNorm.
The residual variant normalizes only its nonlinear branch and has no alpha,
bias, extra output projection, or post-addition normalization.
"""
import math

from torch import nn
from torch.nn import functional as F

from nanochat.gpt import Linear, norm


class NonlinearLMHeadFeatures(nn.Module):
    def __init__(self, width, residual=False):
        super().__init__()
        self.residual = residual
        self.gate = Linear(width, width, bias=False)
        self.up = Linear(width, width, bias=False)
        bound = math.sqrt(3 / width)
        nn.init.uniform_(self.gate.weight, -bound, bound)
        nn.init.uniform_(self.up.weight, -bound, bound)

    def forward(self, h):
        branch_input = norm(h) if self.residual else h
        branch = F.silu(self.gate(branch_input)) * self.up(branch_input)
        return h + branch if self.residual else branch
