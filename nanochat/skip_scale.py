"""Static per-channel scale on the vanilla identity skip only.

FP32 parameters and arithmetic, BF16 residual states. Inductor fuses the
multiply/add/cast when the containing Transformer block is compiled.
"""


def scaled_skip(x, branch, scale):
    return (x.float() * scale + branch.float()).to(x.dtype)
