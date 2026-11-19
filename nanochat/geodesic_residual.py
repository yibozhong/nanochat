"""Split-step spherical Log, Exp and transport, in RMS-radius coordinates."""
import math

import torch

from nanochat.momentum import normalize_residual, state_dtype, tangent_project


def sphere_log_target(h, force, rms_floor):
    """Log_h of the branch direction. h has RMS one and radius sqrt(width)."""
    force = state_dtype(force)
    radial = (h * force).mean(-1, keepdim=True)
    tangent = force - radial * h
    transverse2 = tangent.square().mean(-1, keepdim=True)
    force2 = transverse2 + radial.square()
    # atan2 is invariant to the original force magnitude. Fade the log only
    # below the existing force floor, where the target direction disappears.
    strength = (force2 / rms_floor**2).clamp(max=1.)
    resolution2 = 1e-12 * force2.clamp_min(rms_floor**2)
    transverse = torch.maximum(transverse2, resolution2).sqrt()
    angle = torch.atan2(transverse, radial)
    log = (strength * angle / transverse) * tangent
    # At the cut locus there is no unique shortest arc. Use a deterministic
    # orthogonal direction for indistinguishable antipodes (RoPE width is even).
    left, right = h.chunk(2, dim=-1)
    antipodal_log = (math.pi * strength) * torch.cat((-right, left), dim=-1)
    return torch.where((radial < 0) & (transverse2 <= resolution2), antipodal_log, log)


def sphere_exp_transport(h, step, memory):
    """Return Exp_h(step) and parallel transport of memory along that same arc."""
    angle2 = step.square().mean(-1, keepdim=True)
    angle = angle2.clamp_min(1e-8).sqrt()
    half = .5 * angle
    # sinc = sin(a)/a and cosc = (cos(a)-1)/a², including finite derivatives at rest.
    small = angle2 < 1e-8
    sinc = torch.where(small, 1 - angle2/6, angle.sin()/angle)
    cosc = torch.where(small, -.5 + angle2/24, -.5 * (half.sin()/half).square())
    new_h = normalize_residual((1 + angle2*cosc)*h + sinc*step)
    along = (memory * step).mean(-1, keepdim=True)
    carried = memory + along * (cosc*step - sinc*h)
    return new_h, tangent_project(new_h, carried)


def geodesic_split_write(h, v, force, perpendicular, parallel, decay, rms_floor):
    h, v = state_dtype(h), state_dtype(v)
    log = sphere_log_target(h, force, rms_floor)
    along = (log*v).mean(-1, keepdim=True) / v.square().mean(-1, keepdim=True).clamp_min(1e-12)
    step = perpendicular*log + ((parallel-perpendicular)*along)*v
    memory = decay*v + (1-decay)*step
    return sphere_exp_transport(h, step, memory)
