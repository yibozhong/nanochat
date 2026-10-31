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
    if model.config.variant == 'baseline':
        return
    options = asdict(model.config)
    options.update(variant='baseline', fused_connection_grad=False)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(seed)
        reference = ConnectionModel(ConnectionConfig(**options))
    missing, unexpected = model.load_state_dict(reference.state_dict(), strict=False)
    assert not unexpected
    assert all('_connection.' in name or name.startswith('stream_head.') for name in missing)
