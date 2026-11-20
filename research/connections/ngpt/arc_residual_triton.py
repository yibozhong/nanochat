"""Use the shared spherical write kernels for the isotropic Arc benchmark."""
import torch

from nanochat.chord_triton import fused_write


def compile_arc_step(block):
    if any(s.kappa is not None or s.raw_parallel is not None for s in (block.attn_residual, block.mlp_residual)):
        raise ValueError('The benchmark kernel implements the original isotropic Arc write')
    def update(site, h, v, force):
        return fused_write(h, v, force, site.alpha(), site.decay, site.rms_floor)

    def step(h, v, cos, sin, cache=None):
        attention = block.attn_residual
        query = attention.query(h, v).to(cos.dtype)
        h, v = update(attention, h, v, block.attention_branch(query, cos, sin, cache))
        ffn = block.mlp_residual
        query = ffn.query(h, v).to(cos.dtype)
        return update(ffn, h, v, block.mlp_branch(query))

    return torch.compile(step, fullgraph=True, dynamic=False, options={'emulate_precision_casts': True})
