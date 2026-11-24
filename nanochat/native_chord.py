"""Chord residual updates around the native GPT's unchanged Attention and MLP."""
import torch
from torch import nn

from nanochat.lookahead import LookaheadResidual
from nanochat.momentum import normalize_residual, tangent_project


def enter_sphere(mixed, velocity):
    # Native x0 reinjection moves the base point before the next Chord read.
    h = normalize_residual(mixed.float())
    return h.to(mixed.dtype), tangent_project(h, velocity.float()).to(mixed.dtype)


class NativeChordLayer(nn.Module):
    def __init__(self, config, terminal):
        super().__init__()
        self.token_controls = config.native_chord_write in ('token_vnorm', 'token_bounded')
        if self.token_controls:
            from nanochat.chord_token_controls import TokenControlledSite
            self.attention = TokenControlledSite(config)
            self.ffn = TokenControlledSite(config, velocity_consumed=not terminal)
        else:
            split = config.native_chord_write == 'three_scalar'
            self.attention = LookaheadResidual(config, split_velocity=split)
            self.ffn = LookaheadResidual(config, velocity_consumed=not terminal, split_velocity=split)
        self.terminal = terminal
        self.fuse_norm = config.fuse_chord_read_norm
        self.enter = enter_sphere

    def forward(self, block, velocity, mixed, ve, cos_sin, window_size, kv_cache):
        h, v = self.enter(mixed, velocity)
        if self.token_controls:
            q, controls = self.attention.read(h, v)
        else:
            q = self.attention.branch_input(h, v, mixed.dtype, self.fuse_norm)
        attn = block.compiled_attention if kv_cache is None else None
        force = (attn or block.attn)(q, ve, cos_sin, window_size, kv_cache)
        if self.token_controls:
            h, v = self.attention.write(h, v, force, controls)
            q, controls = self.ffn.read(h, v)
        else:
            h, v = self.attention(h, v, force)
            q = self.ffn.branch_input(h, v, mixed.dtype, self.fuse_norm)
        mlp = block.compiled_mlp if kv_cache is None else None
        force = (mlp or block.mlp)(q)
        if self.token_controls:
            return self.ffn.write(h, v, force, controls)
        if self.terminal and h.is_cuda and h.size(-1) in (128, 512):
            from nanochat.chord_triton import split_boundary
            return split_boundary(self.ffn, h, v, force, None), None
        return self.ffn(h, v, force)


class NativeChord(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(NativeChordLayer(config, i == config.depth - 1)
                                    for i in range(config.depth))
        self.velocity_embedding = nn.Embedding(config.vocab_size, config.width, device='meta')

    def prepare(self, config, native, device, dtype, compile_blocks):
        self.layers.to(device=device)
        self.velocity_embedding.to_empty(device=device).to(dtype=dtype)
        nn.init.normal_(self.velocity_embedding.weight, std=config.velocity_embedding_std)
        if compile_blocks:
            enter = torch.compile(enter_sphere, dynamic=False, fullgraph=True)
            for layer, block in zip(self.layers, native.transformer.h):
                layer.enter = enter
                block.compiled_attention = torch.compile(block.attn.forward, dynamic=False, fullgraph=True)
                block.compiled_mlp = torch.compile(block.mlp.forward, dynamic=False, fullgraph=True)

    @torch.no_grad()
    def metrics(self):
        result = {}
        for i, layer in enumerate(self.layers):
            for name, site in [('attn', layer.attention), ('ffn', layer.ffn)]:
                if layer.token_controls:
                    from nanochat.chord_token_controls import CONTROL_NAMES
                    if site.eval_control_moments is not None:
                        std, mean = (x.tolist() for x in site.eval_control_moments)
                        for j, control in enumerate(CONTROL_NAMES[:site.token_weight.shape[0]]):
                            prefix = f'chord/layer_{i}/{name}/{control}'
                            result[f'{prefix}_token_mean'] = mean[j]
                            result[f'{prefix}_token_std'] = std[j]
                    continue
                if site.raw_beta is None:
                    continue
                prefix = f'chord/layer_{i}/{name}'
                a = site.beta().item()
                b, c = site.writes.tolist() if site.writes is not None else (1 - a, 1 - a)
                result.update({f'{prefix}/history': a, f'{prefix}/parallel_write': b,
                               f'{prefix}/perpendicular_write': c})
        return result
