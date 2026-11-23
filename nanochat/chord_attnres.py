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


def _attention_write_normalized(block, h, v, read, cos, sin, cache):
    return block.attn_residual(h, v, block.attn(read, None, (cos, sin), (-1, 0), cache))


def _mlp_write_normalized(block, h, v, read):
    return block.mlp_residual(h, v, block.mlp(read))


class HistoryCorrection(nn.Module):
    def __init__(self, width, entries):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(1))
        self.scorer = AttnResMix(width) if entries > 1 else None

    def forward(self, history, packed, shape):
        if self.scorer is None:
            memory = history[:, 0] if packed else history[0]
        elif packed:
            memory = self.scorer(history[:, -1], history[:, :-1])
        else:
            from nanochat.chord_attnres_triton import mix_endpoints
            memory = mix_endpoints(history[-1], history[:-1], self.scorer)
        return memory.reshape(shape)


class ChordAttnResModel(LookaheadModel):
    def __init__(self, config):
        super().__init__(config)
        # Before the first completed block q is the only candidate. Both the read
        # and its backward are identity, so there are no score parameters there.
        self.first_mix = min(config.attnres_block_layers, config.depth)
        self.depth_reads = nn.ModuleList()
        for i in range(self.first_mix, config.depth):
            if config.chord_history_correction:
                readers = [HistoryCorrection(config.width, i // config.attnres_block_layers) for _ in range(2)]
            else:
                readers = [AttnResMix(config.width) for _ in range(2)]
            self.depth_reads.append(nn.ModuleList(readers))
        self.attention_write = _attention_write_normalized if config.chord_history_correction else _attention_write
        self.mlp_write = _mlp_write_normalized if config.chord_history_correction else _mlp_write
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
            # Compile the actual functions, then bind the blocks. Compiling partials
            # puts unrelated read/write variants in Dynamo's same wrapper-code cache.
            attention = torch.compile(self.attention_write, **options)
            mlp = torch.compile(self.mlp_write, **options)
            self.compiled_mixed = [(partial(attention, b), partial(mlp, b))
                                   for b in self.blocks[self.first_mix:]]
        else:
            self.compiled_mixed = None
        return self

    def _mixed_step(self, index, h, v, history, cos, sin, kv_cache):
        b = self.blocks[index]
        readers = self.depth_reads[index - self.first_mix]
        if self.compiled_mixed is not None and kv_cache is None:
            attention, mlp = self.compiled_mixed[index - self.first_mix]
        else:
            attention, mlp = partial(self.attention_write, b), partial(self.mlp_write, b)
        read = self._read(readers[0], b.attn_residual, h, v, history)
        h, v = attention(h, v, read, cos, sin, kv_cache)
        read = self._read(readers[1], b.mlp_residual, h, v, history)
        return mlp(h, v, read)

    def _read(self, reader, site, h, v, history):
        if not self.config.chord_history_correction:
            return self._mix(reader, site.query(h, v), history)
        memory = reader(history, self.config.chord_attnres_backend == 'packed', h.shape)
        if h.is_cuda and self.config.fuse_chord_read_norm:
            from nanochat.chord_triton import corrected_query
            return corrected_query(h, v, site.mu, memory, reader.gain)
        q = site.query(h, v)
        read = (q.float() + reader.gain * memory.float()).to(q.dtype)
        return torch.nn.functional.rms_norm(read, (read.size(-1),))

    @torch.no_grad()
    def residual_metrics(self):
        metrics = super().residual_metrics()
        if self.config.chord_history_correction and len(self.depth_reads):
            gains = torch.cat([r.gain for layer in self.depth_reads for r in layer]).tolist()
            for i, gain in enumerate(gains, start=2*self.first_mix):
                metrics[f'residual/sublayer_{i}/history_gain'] = gain
        return metrics

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
        block_start = h if self.config.chord_history_updates else None
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
                value = h
                if block_start is not None:
                    # Subtract the actual stored states after spherical writes.
                    # Both endpoints retain their autograd edges.
                    value = h - block_start
                    block_start = h
                if self.config.chord_attnres_backend == 'packed':
                    value = value.flatten(0, 1).unsqueeze(1)
                    history = value if i+1 == self.first_mix else torch.cat((history, value), dim=1)
                else:
                    history = (*history, value)
        return (h, torch.stack(states, dim=2)) if states is not None else h
