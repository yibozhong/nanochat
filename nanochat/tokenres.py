"""TokenRes: replace the identity path with causal local cross-attention."""
import math
from torch import nn
from nanochat.gpt import Linear, norm
from nanochat import flash_attention


class TokenResidual(nn.Module):
    def __init__(self, width, heads, window=128):
        super().__init__()
        self.width, self.heads, self.head_dim = width, heads, width // heads
        self.window = window
        self.q = Linear(width, width, bias=False)
        self.k = Linear(width, width, bias=False)
        self.v = Linear(width, width, bias=False)
        self.o = Linear(width, width, bias=False)
        for layer in (self.q, self.k, self.v, self.o):
            # Preserve the previous nonzero initialization while isolating the
            # normalization change. This is not a runtime gate.
            bound = math.sqrt(3 / width) * (0.1 if layer is self.o else 1.0)
            nn.init.uniform_(layer.weight, -bound, bound)

    def forward(self, x, u):
        # Match the backbone's pre-RMSNorm and per-head QK norm/scale. Q reads
        # the new branch, K/V read the old state. There is no identity bypass.
        if not flash_attention.USE_FA3:
            raise RuntimeError('TokenRes requires FA3 sliding-window kernels; no dense fallback')
        b, t, d = x.shape
        xn, un = norm(x), norm(u)
        q = self.q(un).view(b, t, self.heads, self.head_dim)
        k = self.k(xn).view(b, t, self.heads, self.head_dim)
        v = self.v(xn).view(b, t, self.heads, self.head_dim)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        r = flash_attention.flash_attn_func(q, k, v, causal=True,
            window_size=(self.window - 1, 0))
        return u + self.o(r.reshape(b, t, d))
