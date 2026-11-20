"""Load version-pinned NVIDIA TE mHC operations and register its TMA allocator."""
from functools import lru_cache
from importlib.metadata import version

import torch


@lru_cache(maxsize=1)
def _load_nvidia_ops():
    if version('transformer-engine') != '2.20.2':
        raise RuntimeError('NVIDIA projection requires the pinned TE 2.20.2 environment')
    import transformer_engine.pytorch.triton.mhc as ops
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
