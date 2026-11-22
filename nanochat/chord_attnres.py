"""Block-endpoint depth attention on the read side of scalar Chord."""
from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from nanochat.attention_residuals import AttnResMix
from nanochat.lookahead import LookaheadModel


def _attention_write(block, h, v, read, cos, sin, cache):
    return block.attn_residual(h, v, block.attention_branch(read, cos, sin, cache))


def _mlp_write(block, h, v, read):
    return block.mlp_residual(h, v, block.mlp_branch(read))


class ChordAttnResModel(LookaheadModel):
    def __init__(self, config):
        super().__init__(config)
        # Before the first completed block q is the only candidate. Both the read
        # and its backward are identity, so there are no score parameters there.
        self.first_mix = min(config.attnres_block_layers, config.depth)
        self.depth_reads = nn.ModuleList([
            nn.ModuleList([AttnResMix(config.width), AttnResMix(config.width)])
            for _ in range(self.first_mix, config.depth)
        ])
        self.compiled_mixed = None

    def residual_parameters(self):
        return self.depth_reads.parameters()

    def prepare(self, device='cuda', dtype=torch.bfloat16, compile_blocks=False, **kwargs):
        if kwargs.get('chord_kernel', 'current') != 'current':
            raise ValueError('ChordAttnRes uses the current Chord write kernel')
        super().prepare(device, dtype, compile_blocks, **kwargs)
        if compile_blocks:
            # Compile branches independently of the number of history entries.
            options = dict(dynamic=False, fullgraph=True, options={'emulate_precision_casts': True})
            self.compiled_mixed = [
                (torch.compile(partial(_attention_write, b), **options),
                 torch.compile(partial(_mlp_write, b), **options))
                for b in self.blocks[self.first_mix:]
            ]
        else:
            self.compiled_mixed = None
        return self

    def _mixed_step(self, index, h, v, history, cos, sin, kv_cache):
        b = self.blocks[index]
        readers = self.depth_reads[index - self.first_mix]
        if self.compiled_mixed is not None and kv_cache is None:
            attention, mlp = self.compiled_mixed[index - self.first_mix]
        else:
            attention, mlp = partial(_attention_write, b), partial(_mlp_write, b)
        read = self._mix(readers[0], b.attn_residual.query(h, v), history)
        h, v = attention(h, v, read, cos, sin, kv_cache)
        read = self._mix(readers[1], b.mlp_residual.query(h, v), history)
        return mlp(h, v, read)

    def _mix(self, reader, q, history):
        if self.config.chord_attnres_backend == 'packed':
            return reader(q, history)
        from nanochat.chord_attnres_triton import mix_endpoints
        return mix_endpoints(q, history, reader)

    def backbone(self, x, kv_cache=None, return_layers=False, token_ids=None):
        start = 0 if kv_cache is None else kv_cache.get_pos()
        if start + x.shape[1] > self.config.sequence_length:
            raise ValueError('Input exceeds configured context length')
        remat = self.config.activation_checkpoint and self.training and torch.is_grad_enabled()
        if remat and kv_cache is not None:
            raise ValueError('Activation checkpointing cannot mutate a KV cache')
        h, v = self.initial_state(x, token_ids, start)
        cos, sin = self.cos[:, start:start+x.shape[1]], self.sin[:, start:start+x.shape[1]]
        history = ()
        states = [] if return_layers else None
        for i, b in enumerate(self.blocks):
            if i < self.first_mix:
                fn = self.compiled_steps[i] if self.compiled_steps is not None and kv_cache is None else b.momentum_forward
                args = (h, v, cos, sin, kv_cache)
            else:
                fn = partial(self._mixed_step, i)
                args = (h, v, history, cos, sin, kv_cache)
            h, v = checkpoint(fn, *args, use_reentrant=False) if remat else fn(*args)
            if states is not None:
                states.append(h)
            if (i+1) % self.config.attnres_block_layers == 0 and i+1 < self.config.depth:
                if self.config.chord_attnres_backend == 'packed':
                    endpoint = h.flatten(0, 1).unsqueeze(1)
                    history = endpoint if i+1 == self.first_mix else torch.cat((history, endpoint), dim=1)
                else:
                    history = (*history, h)
        return (h, torch.stack(states, dim=2)) if states is not None else h
