"""Explicit initialization and attention policy for matched connection training."""
from dataclasses import asdict

import torch

from nanochat.connection_model import ConnectionConfig, ConnectionModel


def resolve_attention_backend(backend):
    if backend != 'hardware_auto':
        return backend
    major, minor = torch.cuda.get_device_capability()
    if major == 9:
        return 'fa3'
    if (major, minor) == (8, 0) and 'A100' in torch.cuda.get_device_name():
        return 'fa2'
    raise RuntimeError('hardware_auto currently supports Hopper and A100 only')


def configure_attention(backend):
    backend = resolve_attention_backend(backend)
    if backend == 'auto':
        return backend
    if backend == 'fa3':
        from nanochat import flash_attention
        if not flash_attention.HAS_FA3:
            raise RuntimeError('FA3 was explicitly requested but could not load; refusing SDPA fallback')
        flash_attention.USE_FA3 = True
        return backend
    if backend not in ('sdpa_cudnn', 'fa2'):
        raise ValueError(backend)
    from nanochat import flash_attention
    flash_attention.USE_FA3 = False
    torch.backends.cuda.enable_flash_sdp(backend == 'fa2')
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(backend == 'sdpa_cudnn')
    return backend


@torch.no_grad()
def match_vanilla_initialization(model, seed):
    """Copy one independently seeded vanilla backbone; preserve connector init.

    Connection modules consume RNG during construction. Equal global seeds
    alone therefore do not give the two architectures identical backbone
    weights. This explicit copy makes the formal quality comparison matched.
    """
    if model.config.variant == 'baseline':
        return
    if model.config.architecture == 'native':
        # Native initialization finishes the entire backbone before drawing
        # connector weights, so the same seed already matches all parameters.
        return
    options = asdict(model.config)
    options.update(variant='baseline', fused_connection_grad=False, compact_routing=False)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(seed)
        reference = ConnectionModel(ConnectionConfig(**options))
    missing, unexpected = model.load_state_dict(reference.state_dict(), strict=False)
    assert not unexpected
    assert all('_connection.' in name or name.startswith(('stream_head.', 'depth_reads.', 'output_attnres.')) for name in missing)
