# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Autograd and reference mix adapted from NVIDIA-NeMo/Automodel kimi_k3/situ.py.
# Provenance and license: research/connections/attnres/.
"""Block Attention Residuals with NVIDIA's unchanged Triton math.

Block size counts complete Transformer layers (attention + MLP), as in NeMo.
History stores the embedding and completed block sums, never mixed readouts.
Only the current token's history is read. No cross-token residual mixing.
"""
from dataclasses import replace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from nanochat.attnres_triton import attn_res_fwd_triton, attn_res_bwd_triton
from nanochat.connection_model import ConnectionModel
from nanochat.gpt import norm


class _AttnResTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prefix_sum, block_residual, norm_weight, proj_weight, eps):
        prefix_sum = prefix_sum if prefix_sum.stride(-1) == 1 else prefix_sum.contiguous()
        block_residual = block_residual if block_residual.stride(-1) == 1 else block_residual.contiguous()
        out, stats = attn_res_fwd_triton(prefix_sum, block_residual, norm_weight, proj_weight, eps)
        ctx.save_for_backward(prefix_sum, block_residual, norm_weight, proj_weight, stats)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        prefix_sum, block_residual, norm_weight, proj_weight, stats = ctx.saved_tensors
        grad_out = grad_out if grad_out.stride(-1) == 1 else grad_out.contiguous()
        d_prefix, d_block, d_sw = attn_res_bwd_triton(
            prefix_sum, block_residual, norm_weight, proj_weight, grad_out, stats)
        d_norm = (d_sw * proj_weight.float()).to(norm_weight.dtype) if ctx.needs_input_grad[2] else None
        d_proj = (d_sw * norm_weight.float()).to(proj_weight.dtype) if ctx.needs_input_grad[3] else None
        return d_prefix, d_block, d_norm, d_proj, None


def reference_mix(prefix, history, norm_weight, proj_weight, eps=1e-6):
    """NeMo FP32 reference, for validation only; training always uses Triton."""
    values = torch.cat((history, prefix.unsqueeze(1)), dim=1).float()
    keys = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    scores = (keys * (norm_weight.float() * proj_weight.float())).sum(-1)
    return (scores.softmax(-1).unsqueeze(-1) * values).sum(1).to(prefix.dtype)


class AttnResMix(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(width))
        self.proj_weight = nn.Parameter(torch.empty(width))
        # NVIDIA K3 initialization, not a claim of baseline-equivalent averaging.
        nn.init.normal_(self.proj_weight, mean=0., std=.02)
        self.eps = 1e-6

    def forward(self, prefix, history):
        shape = prefix.shape
        out = _AttnResTritonFunction.apply(
            prefix.reshape(-1, shape[-1]), history,
            self.norm_weight, self.proj_weight, self.eps)
        return out.view(shape)


class AttnResModel(ConnectionModel):
    def __init__(self, config):
        # Construct the exact vanilla backbone first: extra parameters do not
        # perturb its RNG/init. The original layers remain independently learned.
        super().__init__(replace(config, variant='baseline'))
        self.config = config
        self.depth_reads = nn.ModuleList([
            nn.ModuleList([AttnResMix(config.width), AttnResMix(config.width)])
            for _ in range(config.depth)])
        self.output_attnres = AttnResMix(config.width)

    def prepare(self, device='cuda', dtype=torch.bfloat16, compile_blocks=False):
        if dtype != torch.bfloat16:
            raise ValueError('AttnRes training uses BF16 activations')
        self.to(device)
        self.wte.to(dtype=dtype)
        self.cos = self.cos.to(dtype); self.sin = self.sin.to(dtype)
        if compile_blocks:
            for block in self.blocks:
                block.compiled_attention = torch.compile(block.attention_branch, dynamic=False, fullgraph=True)
                block.compiled_mlp = torch.compile(block.mlp_branch, dynamic=False, fullgraph=True)
        return self

    def residual_parameters(self):
        yield from self.depth_reads.parameters()
        yield from self.output_attnres.parameters()

    def _layer(self, prefix, history, cos, sin, index, kv_cache=None):
        block = self.blocks[index]
        read_attn, read_mlp = self.depth_reads[index]
        # With empty history the mix is exactly prefix (zero score gradients).
        # Still run it, keeping the first query/norm in the autograd graph.
        h = read_attn(prefix, history)
        boundary = index % self.config.attnres_block_size == 0
        if boundary:
            history = torch.cat((history, prefix.flatten(0, 1).unsqueeze(1)), dim=1)
        delta = (block.compiled_attention or block.attention_branch)(h, cos, sin, kv_cache)
        prefix = delta if boundary else prefix + delta
        h = read_mlp(prefix, history)
        delta = (block.compiled_mlp or block.mlp_branch)(h)
        return prefix + delta, history

    def hidden(self, idx, kv_cache=None):
        start = 0 if kv_cache is None else kv_cache.get_pos()
        if start + idx.shape[1] > self.config.sequence_length:
            raise ValueError('Input exceeds configured context length')
        recompute = self.config.activation_checkpoint and self.training and torch.is_grad_enabled()
        if recompute and kv_cache is not None:
            raise ValueError('Activation checkpointing cannot mutate a KV cache')
        prefix = norm(self.wte(idx))
        history = prefix.new_empty((idx.numel(), 0, self.config.width))
        cos, sin = self.cos[:, start:start+idx.shape[1]], self.sin[:, start:start+idx.shape[1]]
        for index in range(self.config.depth):
            if recompute:
                prefix, history = checkpoint(self._layer, prefix, history, cos, sin, index, use_reentrant=False)
            else:
                prefix, history = self._layer(prefix, history, cos, sin, index, kv_cache)
        return norm(self.output_attnres(prefix, history))
