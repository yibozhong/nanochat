"""Dense read/write projections around a wide identity residual state."""
import math
import torch
from torch import nn
from nanochat.gpt import Linear, norm


class WideResidualConnection(nn.Module):
    def __init__(self, width, expansion=4):
        super().__init__()
        self.down_proj = Linear(expansion*width, width, bias=False)
        self.up_proj = Linear(width, expansion*width, bias=False)

    def read(self, state):
        # The only sublayer pre-norm is over the full wide state.
        return self.down_proj(norm(state))

    def write(self, state, branch):
        return state + self.up_proj(branch)


@torch.no_grad()
def initialize_projection(module):
    # Nonzero, variance-preserving dense matrices. Rectangular full-rank
    # parameterization, not a factorization or a constraint during training.
    bound=math.sqrt(3/module.in_features)
    nn.init.uniform_(module.weight, -bound, bound)
