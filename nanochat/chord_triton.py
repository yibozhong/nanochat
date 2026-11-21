"""Fused spherical residual write and VJP. One Triton program owns one token.

Compile-time modes: 0 isotropic, 1 token alpha gates, 2 token beta gate,
3 split steps with shared beta, 4 single step with a parallel-axis correction.
"""
import torch
import triton
import triton.language as tl


# Compile-time velocity policies: transport + projection, projection only, or ambient.
_GEOMETRY = {'parallel_transport': 0, 'project_only': 1, 'ambient': 2}


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
                  D: tl.constexpr, MODE: tl.constexpr, GEOMETRY: tl.constexpr):
    new_h, mixed_inv, mixed_sq, target, force_inv, force_sq = _position(h, v, force, P, FLOOR2, D, MODE)
    bisector = h + new_h
    bisector_sq = _mean(bisector * bisector, D)
    denom = tl.maximum(bisector_sq, 1e-12)
    scale = 2.0 * _mean(v * bisector, D) / denom
    carried = v - scale * bisector if GEOMETRY == 0 else v
    if MODE == 2:
        geometry = _history_alignment(new_h, carried, new_h - h, D)
        beta = tl.sigmoid(tl.load(P + 2) + tl.load(P + 3) * geometry[0])
    elif MODE == 1 or MODE == 3:
        beta = tl.load(P + 2)
    else:
        beta = BETA
    velocity = beta * carried + (1.0 - beta) * (new_h - h)
    radial = _mean(new_h * velocity, D) if GEOMETRY != 2 else 0.
    return (new_h, velocity, radial, bisector, scale, denom, bisector_sq,
            mixed_inv, mixed_sq, target, force_inv, force_sq, carried, beta)


@triton.jit
def _read_values(h, v, mu, D: tl.constexpr, AMBIENT: tl.constexpr):
    offset = mu * v
    read = h + offset
    sq = _mean(read * read, D) if AMBIENT else 1. + _mean(offset * offset, D)
    inv = tl.rsqrt(tl.maximum(sq, 1e-12))
    return read * inv, read, offset, inv, sq


@triton.jit
def _read_vjp(h, v, mu, dq, D: tl.constexpr, AMBIENT: tl.constexpr):
    unit, read, offset, inv, sq = _read_values(h, v, mu, D, AMBIENT)
    if AMBIENT:
        dh = inv * (dq - tl.where(sq >= 1e-12, _mean(dq * unit, D) * unit, 0.))
        d_offset = dh
    else:
        dh = dq * inv
        d_offset = dh - offset * (_mean(dq * read, D) * inv * inv * inv)
    return dh, mu * d_offset, tl.sum(v * d_offset, 0)


@triton.jit
def _query_kernel(H, V, MU, Q, DH, DV, DMU, D: tl.constexpr, AMBIENT: tl.constexpr, BACKWARD: tl.constexpr,
                  NORM: tl.constexpr):
    row = tl.program_id(0)
    offsets = row.to(tl.int64) * D + tl.arange(0, D)
    h = tl.load(H + offsets).to(tl.float32)
    v = tl.load(V + offsets).to(tl.float32)
    mu = tl.load(MU)
    if NORM or not BACKWARD:
        read = _read_values(h, v, mu, D, AMBIENT)[0]
    if NORM:
        # Match the BF16 interface and the default opmath epsilon of F.rms_norm.
        q = read.to(tl.bfloat16).to(tl.float32)
        inv = tl.rsqrt(_mean(q * q, D) + 1.1920928955078125e-7)
        normalized = q * inv
    if BACKWARD:
        dq = tl.load(Q + offsets).to(tl.float32)
        if NORM:
            dq = ((dq - normalized * _mean(dq * normalized, D)) * inv).to(tl.bfloat16).to(tl.float32)
        dh, dv, dmu = _read_vjp(h, v, mu, dq, D, AMBIENT)
        tl.store(DH + offsets, dh)
        tl.store(DV + offsets, dv)
        tl.store(DMU + row, dmu)
    else:
        if NORM:
            tl.store(Q + offsets, normalized)
        else:
            tl.store(Q + offsets, read)


class _BF16Query(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, v, mu, ambient, normalize):
        q = torch.empty_like(h)
        _query_kernel[(h.numel() // h.size(-1),)](h, v, mu, q, q, q, mu, h.size(-1), ambient, False, normalize,
                                                  enable_fp_fusion=False)
        ctx.save_for_backward(h, v, mu)
        ctx.ambient = ambient
        ctx.normalize = normalize
        return q

    @staticmethod
    def backward(ctx, dq):
        h, v, mu = ctx.saved_tensors
        rows = h.numel() // h.size(-1)
        dh, dv, partials = torch.empty_like(h), torch.empty_like(v), mu.new_empty((rows,))
        _query_kernel[(rows,)](h, v, mu, dq.contiguous(), dh, dv, partials, h.size(-1), ctx.ambient, True, ctx.normalize,
                              enable_fp_fusion=False)
        return dh, dv, partials.sum().reshape_as(mu), None, None


def bf16_query(h, v, mu, ambient, normalize=False):
    return _BF16Query.apply(h.contiguous(), v.contiguous(), mu, ambient, normalize)


@triton.jit
def _forward(H, V, F, P, NH, NV, BETA: tl.constexpr, FLOOR2: tl.constexpr,
             D: tl.constexpr, MODE: tl.constexpr, Q, TERMINAL: tl.constexpr,
             PREVIEW: tl.constexpr, GEOMETRY: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * D + tl.arange(0, D)
    h = tl.load(H + offsets).to(tl.float32)
    v = tl.load(V + offsets).to(tl.float32)
    force = tl.load(F + offsets).to(tl.float32)
    if TERMINAL:
        position = _position(h, v, force, P, FLOOR2, D, MODE)
        tl.store(NH + offsets, position[0])
    else:
        values = _write_values(h, v, force, P, BETA, FLOOR2, D, MODE, GEOMETRY)
        new_h, velocity, radial = values[0], values[1], values[2]
        new_v = velocity - radial * new_h if GEOMETRY != 2 else velocity
        tl.store(NH + offsets, new_h)
        tl.store(NV + offsets, new_v)
        if PREVIEW:
            h_read = new_h.to(H.dtype.element_ty).to(tl.float32)
            v_read = new_v.to(V.dtype.element_ty).to(tl.float32)
            q = _read_values(h_read, v_read, tl.load(P + 3), D, GEOMETRY == 2)[0]
            tl.store(Q + offsets, q.to(tl.bfloat16, fp_downcast_rounding="rtne"))


@triton.jit
def _backward(H, V, F, P, DNH, DNV, DH, DV, DF, DP, ROWS, BETA: tl.constexpr,
              FLOOR2: tl.constexpr, D: tl.constexpr, MODE: tl.constexpr, DQ, TERMINAL: tl.constexpr,
              PREVIEW: tl.constexpr, GEOMETRY: tl.constexpr):
    row = tl.program_id(0)
    offsets = row.to(tl.int64) * D + tl.arange(0, D)
    h = tl.load(H + offsets).to(tl.float32)
    v = tl.load(V + offsets).to(tl.float32)
    force = tl.load(F + offsets).to(tl.float32)
    dh_next = tl.load(DNH + offsets).to(tl.float32)
    if TERMINAL:
        new_h, mixed_inv, mixed_sq, target, force_inv, force_sq = _position(h, v, force, P, FLOOR2, D, MODE)
        d_new_h = dh_next
        d_h = tl.full((D,), 0., tl.float32)
        d_v = tl.full((D,), 0., tl.float32)
    else:
        (new_h, velocity, radial, bisector, scale, denom, bisector_sq, mixed_inv,
         mixed_sq, target, force_inv, force_sq, carried, beta) = _write_values(h, v, force, P, BETA, FLOOR2, D, MODE, GEOMETRY)
        # Reverse the tangent projection, then the short-arc parallel transport.
        upstream = tl.load(DNV + offsets).to(tl.float32)
        if PREVIEW:
            new_v = velocity - radial * new_h if GEOMETRY != 2 else velocity
            mu = tl.load(P + 3)
            h_read = new_h.to(H.dtype.element_ty).to(tl.float32)
            v_read = new_v.to(V.dtype.element_ty).to(tl.float32)
            dq = tl.load(DQ + offsets).to(tl.float32)
            dh_q, dv_q, dmu = _read_vjp(h_read, v_read, mu, dq, D, GEOMETRY == 2)
            # The preview reads the stored state. Its adjoints cross the same
            # storage casts as a separate query before merging with other users.
            dh_next = (dh_next + dh_q.to(H.dtype.element_ty).to(tl.float32)).to(H.dtype.element_ty).to(tl.float32)
            upstream = (upstream + dv_q.to(V.dtype.element_ty).to(tl.float32)).to(V.dtype.element_ty).to(tl.float32)
            tl.store(DP + 3 * ROWS + row, dmu)
        if GEOMETRY == 2:
            d_velocity = upstream
            d_chord = (1.0 - beta) * d_velocity
            d_new_h = dh_next + d_chord
        else:
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
        if GEOMETRY == 0:
            coeff = -2.0 * _mean(d_carried * bisector, D) / denom
            d_v = d_carried + coeff * bisector
            d_bisector = coeff * (v - tl.where(bisector_sq >= 1e-12, scale * bisector, 0.0)) - scale * d_carried
            d_h = d_bisector - d_chord
            d_new_h += d_bisector
        else:
            d_v = d_carried
            d_h = -d_chord

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


# Keep the backward-only GEMM distinct in AOTAutograd's joint graph. CSE of
# two plain F.linear calls could otherwise retain the forward force tensor.
@torch.library.custom_op("nanochat::rebuild_chord_force", mutates_args=())
def _rebuild_force(features: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(features, weight)


@_rebuild_force.register_fake
def _fake_rebuild_force(features, weight):
    return features.new_empty((*features.shape[:-1], weight.shape[0]))


class _ChordWrite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, v, force, params, fixed_beta, floor, mode, geometry, *projection):
        width = h.size(-1)
        assert h.is_cuda and h.dtype == v.dtype and h.dtype in (torch.bfloat16, torch.float32)
        assert params.dtype == torch.float32
        assert h.shape == v.shape == force.shape and params.numel() == (1, 5, 4, 3, 2)[mode]
        assert width == triton.next_power_of_2(width) and force.dtype in (torch.bfloat16, torch.float32)
        new_h, new_v = torch.empty_like(h), torch.empty_like(v)
        rows = h.numel() // width
        _forward[(rows,)](h, v, force, params, new_h, new_v, fixed_beta, floor**2,
                          width, mode, params, False, False, geometry, num_warps=4, enable_fp_fusion=False)
        ctx.remat = bool(projection)
        ctx.save_for_backward(h, v, params, *(projection or (force,)))
        ctx.options = fixed_beta, floor**2, width, mode
        ctx.geometry = geometry
        return new_h, new_v

    @staticmethod
    def backward(ctx, d_new_h, d_new_v):
        h, v, params, *saved = ctx.saved_tensors
        if ctx.remat:
            # Reuse the original linear node for its VJP. Only its value is
            # needed here, using the same BF16 activation and cast weight.
            with torch.no_grad():
                force = _rebuild_force(saved[0], saved[1])
        else:
            force, = saved
        rows = h.numel() // h.size(-1)
        dh, dv, df = torch.empty_like(h), torch.empty_like(v), torch.empty_like(force)
        partials = params.new_empty((params.numel(), rows))
        _backward[(rows,)](h, v, force, params, d_new_h.contiguous(), d_new_v.contiguous(),
                           dh, dv, df, partials, rows, *ctx.options, params, False, False, ctx.geometry, num_warps=4, enable_fp_fusion=False)
        grads = (dh, dv, df, partials.sum(1).reshape_as(params), None, None, None, None)
        return grads + ((None, None) if ctx.remat else ())


class _SplitBoundary(torch.autograd.Function):
    """Two parameters for a terminal write, or [perp, parallel, beta, next_mu] for a preview."""
    @staticmethod
    def forward(ctx, h, v, force, params, floor, terminal, geometry):
        width = h.size(-1)
        nh = torch.empty_like(h)
        nv = params if terminal else torch.empty_like(v)
        q = params if terminal else torch.empty_like(h, dtype=torch.bfloat16)
        _forward[(h.numel() // width,)](h, v, force, params, nh, nv, 0., floor**2, width, 3,
                                        q, terminal, not terminal, geometry, num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(h, v, force, params)
        ctx.floor2 = floor**2
        ctx.geometry = geometry
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
        dnv = params if terminal else grads[1].contiguous()
        dq = params if terminal else grads[2].contiguous()
        _backward[(rows,)](h, v, force, params, dnh, dnv, dh, dv, df, partials, rows,
                           0., ctx.floor2, width, 3, dq, terminal, not terminal, ctx.geometry,
                           num_warps=4, enable_fp_fusion=False)
        return dh, dv, df, partials.sum(1), None, None, None


def split_boundary(site, h, v, force, next_mu):
    params = [site.alpha(), site.step_size(site.raw_parallel)]
    if next_mu is not None:
        params.extend((site.beta(), next_mu[0]))
    return _SplitBoundary.apply(h.contiguous(), v.contiguous(), force.contiguous(), torch.stack(params),
                                site.rms_floor, next_mu is None, _GEOMETRY[site.velocity_geometry])


def fused_gated_write(h, v, force, params, floor):
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), params, 0., floor, 1, 0)


def fused_beta_write(h, v, force, params, floor):
    """Positive split steps with per-token velocity decay. P holds both steps, bias and slope."""
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), params, 0., floor, 2, 0)


def fused_write(h, v, force, alpha, beta, floor):
    """Isotropic Chord write."""
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), alpha, beta, floor, 0, 0)


def fused_split_write(h, v, force, params, floor, geometry='parallel_transport', projection=()):
    """P holds the perpendicular and parallel step sizes and the effective scalar beta."""
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), params, 0., floor, 3, _GEOMETRY[geometry], *projection)


def fused_axis_write(h, v, force, params, beta, floor):
    """P holds the isotropic step and kappa; beta is fixed."""
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), params, beta, floor, 4, 0)
