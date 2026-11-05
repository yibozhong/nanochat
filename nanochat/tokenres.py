"""TokenRes: replace the identity path with causal local cross-attention."""
import math
from torch import nn
from nanochat.gpt import Linear
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
            nn.init.uniform_(layer.weight, -math.sqrt(3 / width), math.sqrt(3 / width))

    def forward(self, x, u):
        # Raw x and u, exactly as specified: no extra RMSNorm, RoPE, gate,
        # score bias, or identity bypass. FA supplies 1/sqrt(head_dim).
        if not flash_attention.USE_FA3:
            raise RuntimeError('TokenRes requires FA3 sliding-window kernels; no dense fallback')
        b, t, d = x.shape
        q = self.q(u).view(b, t, self.heads, self.head_dim)
        k = self.k(x).view(b, t, self.heads, self.head_dim)
        v = self.v(x).view(b, t, self.heads, self.head_dim)
        r = flash_attention.flash_attn_func(q, k, v, causal=True,
            window_size=(self.window - 1, 0))
        return u + self.o(r.reshape(b, t, d))
