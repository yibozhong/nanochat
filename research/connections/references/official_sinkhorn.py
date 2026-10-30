# Copyright (c) 2025 DeepSeek. MIT; see DeepSeek-LICENSE.
# Unmodified function from DeepGEMM 5081f8513af57e61696de4f4f04b0e707ffb503e
# tests/test_mega_mhc.py, used as a numerical oracle.
import torch

def _sinkhorn_reference(value: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    value = value.softmax(dim=-1) + eps
    value = value / (value.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        value = value / (value.sum(dim=-1, keepdim=True) + eps)
        value = value / (value.sum(dim=-2, keepdim=True) + eps)
    return value

