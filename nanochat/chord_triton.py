"""Fused spherical residual write and VJP. One Triton program owns one token."""
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
def _write_values(h, v, force, P, BETA: tl.constexpr, FLOOR2: tl.constexpr,
                  D: tl.constexpr, GATED: tl.constexpr):
    force_sq = _mean(force * force, D)
    force_inv = tl.rsqrt(tl.maximum(force_sq, FLOOR2))
    target = force * force_inv
    delta = target - h
    if GATED:
        split = _split(delta, v, P, D)
        perp, par, along = split[0], split[1], split[2]
        mixed = (h + perp * delta) + ((par - perp) * along) * v
        beta = tl.load(P + 2)
    else:
        mixed = h + tl.load(P) * delta
        beta = BETA
    mixed_sq = _mean(mixed * mixed, D)
    mixed_inv = tl.rsqrt(tl.maximum(mixed_sq, 1e-12))
    new_h = mixed * mixed_inv
    bisector = h + new_h
    bisector_sq = _mean(bisector * bisector, D)
    denom = tl.maximum(bisector_sq, 1e-12)
    scale = 2.0 * _mean(v * bisector, D) / denom
    carried = v - scale * bisector
    velocity = beta * carried + (1.0 - beta) * (new_h - h)
    radial = _mean(new_h * velocity, D)
    return (new_h, velocity, radial, bisector, scale, denom, bisector_sq,
            mixed_inv, mixed_sq, target, force_inv, force_sq, carried)


@triton.jit
def _forward(H, V, F, P, NH, NV, BETA: tl.constexpr, FLOOR2: tl.constexpr,
             D: tl.constexpr, GATED: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * D + tl.arange(0, D)
    h, v = tl.load(H + offsets), tl.load(V + offsets)
    force = tl.load(F + offsets).to(tl.float32)
    values = _write_values(h, v, force, P, BETA, FLOOR2, D, GATED)
    new_h, velocity, radial = values[0], values[1], values[2]
    tl.store(NH + offsets, new_h)
    tl.store(NV + offsets, velocity - radial * new_h)


@triton.jit
def _backward(H, V, F, P, DNH, DNV, DH, DV, DF, DP, ROWS, BETA: tl.constexpr,
              FLOOR2: tl.constexpr, D: tl.constexpr, GATED: tl.constexpr):
    row = tl.program_id(0)
    offsets = row.to(tl.int64) * D + tl.arange(0, D)
    h, v = tl.load(H + offsets), tl.load(V + offsets)
    force = tl.load(F + offsets).to(tl.float32)
    (new_h, velocity, radial, bisector, scale, denom, bisector_sq, mixed_inv,
     mixed_sq, target, force_inv, force_sq, carried) = _write_values(h, v, force, P, BETA, FLOOR2, D, GATED)
    if GATED:
        beta = tl.load(P + 2)
    else:
        beta = BETA
    # Reverse the tangent projection, then the short-arc parallel transport.
    upstream = tl.load(DNV + offsets)
    dot = _mean(upstream * new_h, D)
    d_velocity = upstream - dot * new_h
    d_new_h = tl.load(DNH + offsets) + ((1.0 - beta) * d_velocity - radial * upstream - dot * velocity)
    d_carried = beta * d_velocity
    coeff = -2.0 * _mean(d_carried * bisector, D) / denom
    d_v = d_carried + coeff * bisector
    d_bisector = coeff * (v - tl.where(bisector_sq >= 1e-12, scale * bisector, 0.0)) - scale * d_carried
    d_h = d_bisector - (1.0 - beta) * d_velocity
    d_new_h += d_bisector

    dot_h = _mean(d_new_h * new_h, D)
    d_mixed = mixed_inv * (d_new_h - tl.where(mixed_sq >= 1e-12, new_h * dot_h, 0.0))
    delta = target - h
    if GATED:
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
    def forward(ctx, h, v, force, params, fixed_beta, floor, gated):
        width = h.size(-1)
        assert h.is_cuda and h.dtype == v.dtype == params.dtype == torch.float32
        assert h.shape == v.shape == force.shape and params.numel() == (5 if gated else 1)
        assert width == triton.next_power_of_2(width) and force.dtype in (torch.bfloat16, torch.float32)
        new_h, new_v = torch.empty_like(h), torch.empty_like(v)
        rows = h.numel() // width
        _forward[(rows,)](h, v, force, params, new_h, new_v, fixed_beta, floor**2,
                          width, gated, num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(h, v, force, params)
        ctx.options = fixed_beta, floor**2, width, gated
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


def fused_gated_write(h, v, force, params, floor):
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), params, 0., floor, True)


def fused_write(h, v, force, alpha, beta, floor):
    """Original single-step-size Arc benchmark entry point."""
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), alpha, beta, floor, False)
