"""Parameter-free, per-token depth memory in the existing FFN coordinates."""
import torch
import torch.nn.functional as F


def ffn_width_residual(z, state):
    """Read OLD state at z's RMS, then accumulate the original z.

    Both inputs have shape [B, T, M]. Statistics use FP32 for BF16 inputs;
    mixed activations and the out-of-place running sum retain their dtype.
    No detach: both state and the current RMS participate in outer gradients.
    """
    if z.shape != state.shape or z.dtype != state.dtype:
        raise ValueError('FFN activation and depth state must have identical shape/dtype')
    dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
    sf, zf = state.to(dtype), z.to(dtype)
    # vector_norm has a finite zero subgradient, unlike sqrt(mean(z**2)) at 0.
    rms_z = torch.linalg.vector_norm(zf, dim=-1, keepdim=True) / (z.shape[-1] ** .5)
    residual = (F.rms_norm(sf, (sf.shape[-1],), eps=1e-6) * rms_z).to(z.dtype)
    return z + residual, state + z
