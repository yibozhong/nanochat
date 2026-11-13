"""Detached, additive latent feedback from ifm-ai/LoRM.

The nanochat backbone, normalized input embedding and readout are retained.
Training samples P1/P2 once per update; evaluation defaults to P4.
Cached decoding is the exact token recurrence, including prompt prefill.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nanochat.connection_model import ConnectionModel
from nanochat.engine import KVCache
from nanochat.gpt import Linear, norm


def sampled_passes(seed, step):
    """LoRM's stateless draw: identical across microbatches and checkpoint resume."""
    return int(np.random.default_rng([seed, step]).choice(2, p=[.15, .85])) + 1


class FeedbackKVCache(KVCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feedback_state = None

    def reset(self):
        super().reset()
        self.feedback_state = None

    def prefill(self, other):
        if not isinstance(other, FeedbackKVCache):
            raise TypeError('Feedback prefill requires a feedback cache')
        super().prefill(other)
        self.feedback_state = (None if other.feedback_state is None else
            other.feedback_state.expand(self.batch_size, -1, -1).clone())


class LatentFeedbackModel(ConnectionModel):
    def __init__(self, config):
        super().__init__(config)
        # Construct after the baseline so shared tensors have identical seed init.
        self.fusion = Linear(config.width, config.width, bias=False)
        nn.init.normal_(self.fusion.weight, std=.02)
        self.compiled_mix = None

    def mix(self, embedding, top):
        fed = self.fusion(F.rms_norm(top, (top.shape[-1],), eps=1e-5))
        return torch.cat((embedding[:, :1], embedding[:, 1:] + fed[:, :-1]), dim=1)

    def prepare(self, device='cuda', dtype=torch.bfloat16, compile_blocks=False):
        super().prepare(device, dtype, compile_blocks)
        if compile_blocks:
            self.compiled_mix = torch.compile(self.mix, dynamic=False, fullgraph=True)
        return self

    @staticmethod
    def training_passes(seed, step):
        return sampled_passes(seed, step)

    def hidden(self, idx, kv_cache=None, passes=None):
        if idx.ndim != 2 or idx.shape[1] < 1:
            raise ValueError('Expected a nonempty [batch, tokens] input')
        if kv_cache is not None:
            if passes is not None:
                raise ValueError('Cached decoding uses exact recurrence, not parallel passes')
            if torch.is_grad_enabled():
                raise ValueError('Cached feedback decoding requires no_grad/inference_mode')
            if not isinstance(kv_cache, FeedbackKVCache):
                raise TypeError('Use FeedbackKVCache to retain the previous top state')
            if kv_cache.get_pos() + idx.shape[1] > self.config.sequence_length:
                raise ValueError('Input exceeds configured context length')
            if kv_cache.get_pos() and kv_cache.feedback_state is None:
                raise ValueError('Nonempty cache is missing its feedback state')
            outputs = []
            for token in idx.split(1, dim=1):
                x = norm(self.wte(token))
                if kv_cache.get_pos():
                    prev = kv_cache.feedback_state
                    x = x + self.fusion(F.rms_norm(prev, (prev.shape[-1],), eps=1e-5))
                top = self.backbone(x, kv_cache)
                kv_cache.feedback_state = top.detach()
                outputs.append(top)
            return norm(torch.cat(outputs, dim=1))
        passes = (2 if self.training else self.config.feedback_eval_passes) if passes is None else passes
        if type(passes) is not int or passes < 1:
            raise ValueError('passes must be a positive integer')
        embedding = norm(self.wte(idx))
        grad_enabled = torch.is_grad_enabled()
        for index in range(passes):
            with torch.set_grad_enabled(grad_enabled and index == passes - 1):
                state = embedding if index == 0 else (self.compiled_mix or self.mix)(embedding, top.detach())
                top = self.backbone(state)
        return norm(top)

    def forward(self, idx, targets=None, dense=False, kv_cache=None, passes=None):
        return self.readout(self.hidden(idx, kv_cache, passes), targets, dense)
