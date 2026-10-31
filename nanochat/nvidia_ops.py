"""Load hash-pinned NVIDIA TE mHC operations and register its TMA allocator."""
from functools import lru_cache
import hashlib
from importlib.metadata import version
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _load_nvidia_ops():
    if version('transformer-engine') != '2.20.2':
        raise RuntimeError('NVIDIA projection requires the pinned TE 2.20.2 environment')
    import transformer_engine.pytorch.triton.mhc as ops
    import transformer_engine.common.triton.mhc as kernels
    for module, digest in (
        (ops, '4e71eff5d727625ef188e5ec86c2f91b25baf3742b7798313372b2a022e886da'),
        (kernels, '0045990824e46034a2d3162a4690fec1fa87f880dbbab0f43c546ba29d62ff7f'),
    ):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'NVIDIA mHC source mismatch: {module.__file__}')
    return ops


def _allocate_tma_workspace(size, alignment, stream):
    # Same CUDA allocation as TE's initializer; CUDA's allocator provides the
    # required alignment. Registration is per ContextVar in Triton 3.6.
    return torch.empty(size, device='cuda', dtype=torch.int8)


def nvidia_ops():
    ops = _load_nvidia_ops()
    # TE guards initialization with a process-global bool, but Triton 3.6's
    # allocator lives in a ContextVar. Checkpoint recomputation can enter from
    # an autograd worker with a fresh context after that bool is already true.
    # Preserve any valid allocator and register only in an empty context.
    import triton
    from triton.runtime import _allocation
    if isinstance(_allocation._allocator.get(), _allocation.NullAllocator):
        triton.set_allocator(_allocate_tma_workspace)
    return ops
