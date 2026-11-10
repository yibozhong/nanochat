"""Packed-training adapter for the complete native GPT with optional NVIDIA mHC.

GPT.hidden is the single source of truth for SSSL, VE, smear, x0, scaling and
backout. This adapter only provides the experiment harness and full-vocab CCE.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig
from nanochat.nvidia_connections import check_nvidia_device


class NativeConnectionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.architecture != 'native':
            raise ValueError('NativeConnectionModel requires architecture=native')
        self.config = config
        native = GPTConfig(sequence_len=config.sequence_length, vocab_size=config.vocab_size,
            n_layer=config.depth, n_head=config.heads, n_kv_head=config.heads,
            n_embd=config.width, window_pattern=config.window_pattern,
            connection_variant=config.variant, connection_backend=config.connection_backend,
            fused_connection_grad=config.fused_connection_grad,
            activation_checkpoint=config.activation_checkpoint)
        # Match upstream base_train: meta construction, device allocation, then
        # one native initialization. Connector initialization comes last.
        with torch.device('meta'):
            self.native = GPT(native)

    def prepare(self, device='cuda', dtype=torch.bfloat16, compile_blocks=False):
        if dtype != COMPUTE_DTYPE:
            raise ValueError('Native GPT dtype is selected by NANOCHAT_DTYPE at process startup')
        if self.config.variant != 'baseline':
            check_nvidia_device(device)
            if dtype != torch.bfloat16:
                raise ValueError('NVIDIA connections require BF16 activations')
        self.native.to_empty(device=device)
        self.native.init_weights()
        if compile_blocks:
            self.native.compile_branches()
        return self

    def hidden(self, idx, kv_cache=None):
        self.native.config.activation_checkpoint = self.config.activation_checkpoint
        return self.native.hidden(idx, kv_cache)

    def forward(self, idx, targets=None, dense=False, kv_cache=None):
        h = self.hidden(idx, kv_cache)
        w = self.native.lm_head.weight[:self.config.vocab_size].to(h.dtype)
        if targets is None or dense:
            logits = 15 * torch.tanh(F.linear(h, w).float() / 15)
            if targets is None:
                return logits
            return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-1)
        from cut_cross_entropy import linear_cross_entropy
        return linear_cross_entropy(h, w, targets, softcap=15., shift=0,
            reduction='mean', ignore_index=-1, filter_eps=None,
            filter_e_grad=False, filter_c_grad=False, accum_e_fp32=True, accum_c_fp32=True)

    def setup_optimizer(self, batch_tokens=32768):
        scale = math.sqrt(batch_tokens / 2**19)
        counts = self.native.num_scaling_params()
        scaling_params = counts['transformer_matrices'] + counts['lm_head']
        # Same d12 reference and parameter counting as native base_train,
        # including its VE gates, excluding connectors from backbone scaling.
        reference = GPTConfig(n_layer=12, n_embd=768, n_head=6, n_kv_head=6,
                              vocab_size=self.config.vocab_size)
        with torch.device('meta'):
            reference_counts = GPT(reference).num_scaling_params()
        wd = .28 * scale * (reference_counts['transformer_matrices'] + reference_counts['lm_head']) / scaling_params
        opt = self.native.setup_optimizer(unembedding_lr=.008*scale, embedding_lr=.3*scale,
            matrix_lr=.02*scale, scalar_lr=.5*scale, weight_decay=wd)
        ids = [id(p) for g in opt.param_groups for p in g['params']]
        assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in self.parameters()}
        return opt, wd
