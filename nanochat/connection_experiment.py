"""Explicit initialization and attention policy for matched connection training."""
from dataclasses import asdict

import torch

from nanochat.connection_model import ConnectionConfig, ConnectionModel


def configure_attention(backend):
    if backend == 'auto':
        return
    if backend != 'sdpa_cudnn':
        raise ValueError(backend)
    from nanochat import flash_attention
    flash_attention.USE_FA3 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(True)


@torch.no_grad()
def match_vanilla_initialization(model, seed):
    """Copy one independently seeded vanilla backbone; preserve connector init.

    Connection modules consume RNG during construction. Equal global seeds
    alone therefore do not give the two architectures identical backbone
    weights. This explicit copy makes the formal quality comparison matched.
    """
    if model.config.variant in ('baseline','dfwr'):
        return
    if model.config.architecture == 'native':
        # Native initialization finishes the entire backbone before drawing
        # connector weights, so the same seed already matches all parameters.
        return
    options = asdict(model.config)
    options.update(variant='baseline', fused_connection_grad=False, readout='collapse', partition_mixing='none')
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(seed)
        reference = ConnectionModel(ConnectionConfig(**options))
    state = reference.state_dict()
    if model.config.readout == 'dense_expanded':
        # Same backbone/connector initialization; the larger head cannot copy
        # a D-wide matrix. Use an independent, reproducible CPU RNG instead.
        import math
        state.pop('lm_head.weight')
        generator = torch.Generator(device='cpu').manual_seed(seed+1)
        weights = torch.empty(model.lm_head.weight.shape, dtype=torch.float32, device='cpu')
        weights.normal_(std=.001/math.sqrt(model.config.streams), generator=generator)
        model.lm_head.weight.copy_(weights)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    assert all('_connection.' in name or name.startswith(('stream_head.', 'partition_mixer.'))
               or name in ('partition_ids', 'partition_inverse')
               or (name == 'lm_head.weight' and model.config.readout == 'dense_expanded') for name in missing)
