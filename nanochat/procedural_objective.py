"""The released procedural task vocabulary and independent packed examples."""
from contextlib import contextmanager

import torch
from torch.nn import functional as F


def task_logits(hidden, weight, class_ids):
    """Retain the baseline vocabulary projection FLOPs; normalize only selected columns."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    weight = weight.to(flat.dtype)
    selected = torch.cat([F.linear(h, weight).index_select(-1, class_ids) for h in flat.split(512)])
    return (15 * torch.tanh(selected.float() / 15)).view(*hidden.shape[:-1], len(class_ids))


def document_attention(q, k, v, bias, window_size):
    """Compute the full causal triangle, masking scores that cross an example boundary."""
    # PyTorch's efficient SDPA accepts both a document bias and a causal triangle.
    # A view across heads avoids materializing one quadratic bias per head.
    if window_size[0] >= 0 and window_size[0] < q.size(1):
        t = torch.arange(q.size(1), device=q.device)
        bias = bias.masked_fill(t[:, None] - t > window_size[0], -torch.inf)
    if q.size(2) != k.size(2):
        copies = q.size(2) // k.size(2)
        k, v = k.repeat_interleave(copies, dim=2), v.repeat_interleave(copies, dim=2)
    bias = bias.expand(-1, q.size(2), -1, -1)
    output = torch.ops.aten._scaled_dot_product_efficient_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), bias, True, 0., True)[0]
    return output.transpose(1, 2)


@contextmanager
def independent_examples(model, inputs):
    """Reset RoPE and prevent cross-example reads; keep the scope open through backward."""
    core = getattr(model, 'native', model)
    blocks = core.transformer.h if hasattr(model, 'native') else core.blocks
    pad = inputs == model.config.vocab_size - 1
    # Each pad token has its own segment; even a leading boundary cannot be read.
    segment = 2 * pad.cumsum(1) - pad.long()
    start = torch.ones_like(pad)
    start[:, 1:] = segment[:, 1:] != segment[:, :-1]
    t = torch.arange(inputs.size(1), device=inputs.device).expand_as(inputs)
    pos = t - t.masked_fill(~start, 0).cummax(1).values
    cos, sin = core.cos, core.sin
    batch, length = inputs.shape
    # CUTLASS requires aligned strides even for test contexts that are not multiples of eight.
    aligned = (length + 7) // 8 * 8
    bias = torch.zeros((batch, 1, length, aligned), device=inputs.device, dtype=cos.dtype)[..., :length]
    bias.masked_fill_(segment[:, None, :, None] != segment[:, None, None, :], -torch.inf)
    try:
        core.cos, core.sin = cos[0, pos], sin[0, pos]
        core.procedural_starts = start
        for block in blocks:
            block.attn.procedural_bias = bias
        yield
    finally:
        core.cos, core.sin = cos, sin
        core.procedural_starts = None
        for block in blocks:
            block.attn.procedural_bias = None
