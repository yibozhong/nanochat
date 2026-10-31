"""Test-only objectives and numerical comparisons shared by both revisions.

Production CCE casts its classifier and its returned gradient through BF16.
Independent reductions can straddle a BF16 rounding boundary. Use a dense FP32
readout to isolate architecture gradients; also audit the actual production CCE.
Nothing here is imported by the model or training entry point.
"""
import math

import torch
from torch.nn import functional as F


ATOL = 1e-6
RTOL = 3e-4
LOSS_ATOL = 2e-6
METHOD = 'fp32-dense-parity-and-cce-repeat-control-v2'


def evaluate(model, x, y, objective):
    model.zero_grad(set_to_none=True)
    if objective == 'dense_fp32':
        # Keep the complete BF16 transformer/NVIDIA path. Only the readout uses
        # FP32 GEMM/autograd instead of CCE's low-precision atomic reductions.
        hidden = model.hidden(x)
        logits = 15 * torch.tanh(F.linear(hidden.float(), model.lm_head.weight.float()) / 15)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    elif objective == 'production_cce':
        loss = model(x, y)
    else:
        raise ValueError(objective)
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
    return dict(loss=loss.item(), grads={name: p.grad.detach().cpu().clone()
                                      for name, p in model.named_parameters()})


def compare(left, right, objective):
    assert left['grads'].keys() == right['grads'].keys()
    parameters = {}
    sum_squared_error = sum_squared_reference = 0.
    for name, reference in right['grads'].items():
        actual = left['grads'][name].float()
        expected = reference.float()
        delta = actual - expected
        error_norm = delta.norm().item()
        reference_norm = expected.norm().item()
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        # The absolute tolerance must also apply to near-zero gradients. A pure
        # per-parameter relative L2 criterion incorrectly rejects the old code
        # against itself (e.g. cancelling connector-scale gradients).
        mixed_bound = ATOL * math.sqrt(delta.numel()) + RTOL * reference_norm
        parameters[name] = dict(dtype=str(reference.dtype), finite=finite,
            max_abs=delta.abs().max().item(), relative_l2=error_norm / max(reference_norm, 1e-30),
            mixed_l2_ratio=error_norm / mixed_bound,
            elementwise_violations=int((delta.abs() > ATOL + RTOL * expected.abs()).sum()),
            changed_elements=int(torch.count_nonzero(delta)), elements=delta.numel())
        sum_squared_error += error_norm ** 2
        sum_squared_reference += reference_norm ** 2
    global_relative_l2 = math.sqrt(sum_squared_error / max(sum_squared_reference, 1e-60))
    loss_difference = abs(left['loss'] - right['loss'])
    passed = loss_difference <= LOSS_ATOL and all(p['finite'] for p in parameters.values())
    if objective == 'dense_fp32':
        passed &= all(p['elementwise_violations'] == 0 and p['mixed_l2_ratio'] <= 1
                      for p in parameters.values())
    elif objective == 'production_cce':
        # Retain the 3e-4 norm tolerance for the complete production gradient;
        # individual near-zero gradients are checked by the dense objective.
        # Report every CCE parameter, including BF16 elementwise differences.
        passed &= global_relative_l2 <= RTOL
    else:
        raise ValueError(objective)
    return dict(objective=objective, passed=bool(passed), loss_difference=loss_difference,
                global_relative_l2=global_relative_l2, parameters=parameters,
                tolerances=dict(atol=ATOL, rtol=RTOL, loss_atol=LOSS_ATOL))
