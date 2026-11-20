"""Fused spherical residual write and VJP. One Triton program owns one token.

Compile-time modes: 0 isotropic, 1 token alpha gates, 2 token beta gate,
3 split steps with shared beta, 4 single step with a parallel-axis correction.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _mean(x, D: tl.constexpr):
    return tl.sum(x, 0) / D


@triton.jit
def _split(delta, v, P, D: tl.constexpr):
    # P packs the two base steps, beta, and their two independent gate slopes.
    dot = _mean(delta * v, D)
    v2 = _mean(v * v, D)
    delta2 = _mean(delta * delta, D)
    den = tl.maximum(v2, 1e-12)
    inv_norms = tl.rsqrt(den) * tl.rsqrt(tl.maximum(delta2, 1e-12))
    cosine = dot * inv_norms
    perp_gate = 2. * tl.sigmoid(tl.load(P + 3) * cosine)
    par_gate = 2. * tl.sigmoid(tl.load(P + 4) * cosine)
    perp = tl.load(P) * perp_gate
    par = tl.load(P + 1) * par_gate
    along = dot / den
    return perp, par, along, cosine, perp_gate, par_gate, inv_norms, den, v2, delta2


@triton.jit
def _history_alignment(new_h, carried, chord, D: tl.constexpr):
    radial = _mean(new_h * chord, D)
    recent = chord - radial * new_h
    h2 = _mean(carried * carried, D)
    r2 = _mean(recent * recent, D)
    inv_h = tl.rsqrt(tl.maximum(h2, 1e-12))
    inv_r = tl.rsqrt(tl.maximum(r2, 1e-12))
    inv = inv_h * inv_r
    cosine = _mean(carried * recent, D) * inv
    return cosine, recent, radial, h2, r2, inv


@triton.jit
def _position(h, v, force, P, FLOOR2: tl.constexpr,
                  D: tl.constexpr, MODE: tl.constexpr):
    force_sq = _mean(force * force, D)
    force_inv = tl.rsqrt(tl.maximum(force_sq, FLOOR2))
    target = force * force_inv
    delta = target - h
    if MODE == 1:
        split = _split(delta, v, P, D)
        perp, par, along = split[0], split[1], split[2]
        mixed = (h + perp * delta) + ((par - perp) * along) * v
    elif MODE == 4:
        along = _mean(delta * v, D) / tl.maximum(_mean(v * v, D), 1e-12)
        mixed = h + tl.load(P) * (delta + (tl.load(P + 1) * along) * v)
    elif MODE == 2 or MODE == 3:
        perp, par = tl.load(P), tl.load(P + 1)
        along = _mean(delta * v, D) / tl.maximum(_mean(v * v, D), 1e-12)
        mixed = (h + perp * delta) + ((par - perp) * along) * v
    else:
        mixed = h + tl.load(P) * delta
    mixed_sq = _mean(mixed * mixed, D)
    mixed_inv = tl.rsqrt(tl.maximum(mixed_sq, 1e-12))
    new_h = mixed * mixed_inv
    return new_h, mixed_inv, mixed_sq, target, force_inv, force_sq


@triton.jit
def _write_values(h, v, force, P, BETA: tl.constexpr, FLOOR2: tl.constexpr,
                  D: tl.constexpr, MODE: tl.constexpr):
    new_h, mixed_inv, mixed_sq, target, force_inv, force_sq = _position(h, v, force, P, FLOOR2, D, MODE)
    bisector = h + new_h
    bisector_sq = _mean(bisector * bisector, D)
    denom = tl.maximum(bisector_sq, 1e-12)
    scale = 2.0 * _mean(v * bisector, D) / denom
    carried = v - scale * bisector
    if MODE == 2:
        geometry = _history_alignment(new_h, carried, new_h - h, D)
        beta = tl.sigmoid(tl.load(P + 2) + tl.load(P + 3) * geometry[0])
    elif MODE == 1 or MODE == 3:
        beta = tl.load(P + 2)
    else:
        beta = BETA
    velocity = beta * carried + (1.0 - beta) * (new_h - h)
    radial = _mean(new_h * velocity, D)
    return (new_h, velocity, radial, bisector, scale, denom, bisector_sq,
            mixed_inv, mixed_sq, target, force_inv, force_sq, carried, beta)


@triton.jit
def _forward(H, V, F, P, NH, NV, BETA: tl.constexpr, FLOOR2: tl.constexpr,
             D: tl.constexpr, MODE: tl.constexpr, Q=None, TERMINAL: tl.constexpr=False,
             PREVIEW: tl.constexpr=False):
    offsets = tl.program_id(0).to(tl.int64) * D + tl.arange(0, D)
    h, v = tl.load(H + offsets), tl.load(V + offsets)
    force = tl.load(F + offsets).to(tl.float32)
    if TERMINAL:
        position = _position(h, v, force, P, FLOOR2, D, MODE)
        tl.store(NH + offsets, position[0])
    else:
        values = _write_values(h, v, force, P, BETA, FLOOR2, D, MODE)
        new_h, velocity, radial = values[0], values[1], values[2]
        new_v = velocity - radial * new_h
        tl.store(NH + offsets, new_h)
        tl.store(NV + offsets, new_v)
        if PREVIEW:
            offset = tl.load(P + 3) * new_v
            q = (new_h + offset) * tl.rsqrt(1. + _mean(offset * offset, D))
            tl.store(Q + offsets, q.to(tl.bfloat16, fp_downcast_rounding="rtne"))


@triton.jit
def _backward(H, V, F, P, DNH, DNV, DH, DV, DF, DP, ROWS, BETA: tl.constexpr,
              FLOOR2: tl.constexpr, D: tl.constexpr, MODE: tl.constexpr, DQ=None, TERMINAL: tl.constexpr=False,
              PREVIEW: tl.constexpr=False):
    row = tl.program_id(0)
    offsets = row.to(tl.int64) * D + tl.arange(0, D)
    h, v = tl.load(H + offsets), tl.load(V + offsets)
    force = tl.load(F + offsets).to(tl.float32)
    dh_next = tl.load(DNH + offsets)
    if TERMINAL:
        new_h, mixed_inv, mixed_sq, target, force_inv, force_sq = _position(h, v, force, P, FLOOR2, D, MODE)
        d_new_h = dh_next
        d_h = tl.full((D,), 0., tl.float32)
        d_v = tl.full((D,), 0., tl.float32)
    else:
        (new_h, velocity, radial, bisector, scale, denom, bisector_sq, mixed_inv,
         mixed_sq, target, force_inv, force_sq, carried, beta) = _write_values(h, v, force, P, BETA, FLOOR2, D, MODE)
        # Reverse the tangent projection, then the short-arc parallel transport.
        upstream = tl.load(DNV + offsets)
        if PREVIEW:
            new_v = velocity - radial * new_h
            mu = tl.load(P + 3)
            offset = mu * new_v
            inv = tl.rsqrt(1. + _mean(offset * offset, D))
            dq = tl.load(DQ + offsets).to(tl.float32)
            direct = dq * inv
            d_offset = direct - offset * (_mean(dq * (new_h + offset), D) * inv * inv * inv)
            dh_next += direct
            upstream += mu * d_offset
            tl.store(DP + 3 * ROWS + row, tl.sum(new_v * d_offset, 0))
        dot = _mean(upstream * new_h, D)
        d_velocity = upstream - dot * new_h
        d_chord = (1.0 - beta) * d_velocity
        d_new_h = dh_next + (d_chord - radial * upstream - dot * velocity)
        d_carried = beta * d_velocity
        if MODE == 2:
            chord = new_h - h
            cosine, recent, chord_radial, h2, r2, inv = _history_alignment(new_h, carried, chord, D)
            d_logit = _mean(d_velocity * (carried - chord), D) * beta * (1. - beta)
            tl.store(DP + 2 * ROWS + row, D * d_logit)
            tl.store(DP + 3 * ROWS + row, D * d_logit * cosine)
            d_cos = d_logit * tl.load(P + 3)
            d_carried += d_cos * (inv * recent - tl.where(h2 >= 1e-12, cosine / tl.maximum(h2, 1e-12), 0.) * carried)
            d_recent = d_cos * (inv * carried - tl.where(r2 >= 1e-12, cosine / tl.maximum(r2, 1e-12), 0.) * recent)
            d_radial = _mean(d_recent * new_h, D)
            d_projected_chord = d_recent - d_radial * new_h
            d_chord += d_projected_chord
            d_new_h += d_projected_chord - chord_radial * d_recent - d_radial * chord
        coeff = -2.0 * _mean(d_carried * bisector, D) / denom
        d_v = d_carried + coeff * bisector
        d_bisector = coeff * (v - tl.where(bisector_sq >= 1e-12, scale * bisector, 0.0)) - scale * d_carried
        d_h = d_bisector - d_chord
        d_new_h += d_bisector

    dot_h = _mean(d_new_h * new_h, D)
    d_mixed = mixed_inv * (d_new_h - tl.where(mixed_sq >= 1e-12, new_h * dot_h, 0.0))
    delta = target - h
    if MODE == 1:
        perp, par, along, cosine, gp, gq, inv_norms, den, v2, delta2 = _split(delta, v, P, D)
        dot_v = _mean(d_mixed * v, D)
        d_par = along * dot_v
        d_perp = _mean(d_mixed * delta, D) - d_par
        # Scalar adjoints here use a channel mean. Shared-parameter gradients
        # below use channel sums, followed by a deterministic sum over tokens.
        d_logit_p = d_perp * perp * (1. - .5 * gp)
        d_logit_q = d_par * par * (1. - .5 * gq)
        d_cos = d_logit_p * tl.load(P + 3) + d_logit_q * tl.load(P + 4)
        d_dot_proj = (par - perp) * dot_v / den
        d_dot = d_dot_proj + d_cos * inv_norms
        d_delta = perp * d_mixed + d_dot * v
        d_delta -= tl.where(delta2 >= 1e-12, d_cos * cosine / tl.maximum(delta2, 1e-12), 0.) * delta
        d_v += ((par - perp) * along) * d_mixed + d_dot * delta
        d_v -= tl.where(v2 >= 1e-12, 2. * d_dot_proj * along + d_cos * cosine / den, 0.) * v
        d_h += d_mixed - d_delta
        tl.store(DP + row, D * d_perp * gp)
        tl.store(DP + ROWS + row, D * d_par * gq)
        tl.store(DP + 2 * ROWS + row, tl.sum(d_velocity * (carried - (new_h - h)), 0))
        tl.store(DP + 3 * ROWS + row, D * d_logit_p * cosine)
        tl.store(DP + 4 * ROWS + row, D * d_logit_q * cosine)
    elif MODE == 2 or MODE == 3 or MODE == 4:
        perp = tl.load(P)
        if MODE == 4:
            kappa = tl.load(P + 1)
            gain = perp * kappa
        else:
            gain = tl.load(P + 1) - perp
        v2 = _mean(v * v, D)
        den = tl.maximum(v2, 1e-12)
        along = _mean(delta * v, D) / den
        dot_v = _mean(d_mixed * v, D)
        d_dot = gain * dot_v / den
        d_delta = perp * d_mixed + d_dot * v
        d_v += (gain * along) * d_mixed + d_dot * (delta - tl.where(v2 >= 1e-12, 2. * along * v, 0.))
        d_h += d_mixed - d_delta
        # Contract the normalization Jacobian with the step directions first.
        # For a radial write its angular derivative vanishes; projecting the
        # upstream adjoint first would sum large cancelling terms at zero chord.
        par_dir = along * v
        if MODE == 4:
            base_dir = delta + kappa * par_dir
            axis_dir = perp * par_dir
        else:
            base_dir = delta - par_dir
            axis_dir = par_dir
        if_unclamped = mixed_sq >= 1e-12
        base_dir -= tl.where(if_unclamped, _mean(base_dir * new_h, D) * new_h, 0.)
        axis_dir -= tl.where(if_unclamped, _mean(axis_dir * new_h, D) * new_h, 0.)
        tl.store(DP + row, mixed_inv * tl.sum(d_new_h * base_dir, 0))
        tl.store(DP + ROWS + row, mixed_inv * tl.sum(d_new_h * axis_dir, 0))
        if MODE == 3 and not TERMINAL:
            tl.store(DP + 2 * ROWS + row, tl.sum(d_velocity * (carried - (new_h - h)), 0))
    else:
        alpha = tl.load(P)
        d_h += (1. - alpha) * d_mixed
        d_delta = alpha * d_mixed
        tl.store(DP + row, tl.sum(d_mixed * delta, 0))
    dot_target = _mean(d_delta * target, D)
    d_force = force_inv * (d_delta - tl.where(force_sq >= FLOOR2, target * dot_target, 0.))
    tl.store(DH + offsets, d_h)
    tl.store(DV + offsets, d_v)
    tl.store(DF + offsets, d_force)


class _ChordWrite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, v, force, params, fixed_beta, floor, mode):
        width = h.size(-1)
        assert h.is_cuda and h.dtype == v.dtype == params.dtype == torch.float32
        assert h.shape == v.shape == force.shape and params.numel() == (1, 5, 4, 3, 2)[mode]
        assert width == triton.next_power_of_2(width) and force.dtype in (torch.bfloat16, torch.float32)
        new_h, new_v = torch.empty_like(h), torch.empty_like(v)
        rows = h.numel() // width
        _forward[(rows,)](h, v, force, params, new_h, new_v, fixed_beta, floor**2,
                          width, mode, num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(h, v, force, params)
        ctx.options = fixed_beta, floor**2, width, mode
        return new_h, new_v

    @staticmethod
    def backward(ctx, d_new_h, d_new_v):
        h, v, force, params = ctx.saved_tensors
        rows = h.numel() // h.size(-1)
        dh, dv, df = torch.empty_like(h), torch.empty_like(v), torch.empty_like(force)
        partials = params.new_empty((params.numel(), rows))
        _backward[(rows,)](h, v, force, params, d_new_h.contiguous(), d_new_v.contiguous(),
                           dh, dv, df, partials, rows, *ctx.options, num_warps=4, enable_fp_fusion=False)
        return dh, dv, df, partials.sum(1).reshape_as(params), None, None, None


class _SplitBoundary(torch.autograd.Function):
    """Two parameters for a terminal write, or [perp, parallel, beta, next_mu] for a preview."""
    @staticmethod
    def forward(ctx, h, v, force, params, floor, terminal):
        width = h.size(-1)
        nh = torch.empty_like(h)
        nv = None if terminal else torch.empty_like(v)
        q = None if terminal else torch.empty_like(h, dtype=torch.bfloat16)
        _forward[(h.numel() // width,)](h, v, force, params, nh, nv, 0., floor**2, width, 3,
                                        q, terminal, not terminal, num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(h, v, force, params)
        ctx.floor2 = floor**2
        return nh if terminal else (nh, nv, q)

    @staticmethod
    def backward(ctx, *grads):
        h, v, force, params = ctx.saved_tensors
        width = h.size(-1)
        rows = h.numel() // width
        terminal = len(grads) == 1
        dh, dv, df = torch.empty_like(h), torch.empty_like(v), torch.empty_like(force)
        partials = params.new_empty((params.numel(), rows))
        dnh = grads[0].contiguous()
        dnv = None if terminal else grads[1].contiguous()
        dq = None if terminal else grads[2].contiguous()
        _backward[(rows,)](h, v, force, params, dnh, dnv, dh, dv, df, partials, rows,
                           0., ctx.floor2, width, 3, dq, terminal, not terminal,
                           num_warps=4, enable_fp_fusion=False)
        return dh, dv, df, partials.sum(1), None, None


def split_boundary(site, h, v, force, next_mu):
    params = [site.alpha(), site.step_size(site.raw_parallel)]
    if next_mu is not None:
        params.extend((site.beta(), next_mu[0]))
    return _SplitBoundary.apply(h.contiguous(), v.contiguous(), force.contiguous(), torch.stack(params),
                                site.rms_floor, next_mu is None)


def fused_gated_write(h, v, force, params, floor):
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), params, 0., floor, 1)


def fused_beta_write(h, v, force, params, floor):
    """Positive split steps with per-token velocity decay. P holds both steps, bias and slope."""
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), params, 0., floor, 2)


def fused_write(h, v, force, alpha, beta, floor):
    """Isotropic Chord write."""
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), alpha, beta, floor, 0)


def fused_split_write(h, v, force, params, floor):
    """P holds the perpendicular and parallel step sizes and the effective scalar beta."""
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), params, 0., floor, 3)


def fused_axis_write(h, v, force, params, beta, floor):
    """P holds the isotropic step and kappa; beta is fixed."""
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), params, beta, floor, 4)
