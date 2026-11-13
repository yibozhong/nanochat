"""Stateless auxiliary-loss schedule, indexed by completed optimizer updates."""
import math


def ratio_loss_weight(base_weight, warmup_steps, completed_steps):
    if not math.isfinite(base_weight) or base_weight < 0:
        raise ValueError('Ratio loss weight must be finite and nonnegative')
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError('Warmup steps must be a nonnegative integer')
    if warmup_steps == 0:
        return base_weight
    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError('Warmup training requires the completed optimizer step')
    return base_weight * min(completed_steps / warmup_steps, 1.)
