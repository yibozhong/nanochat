"""Fused spherical residual write and VJP. The forward owns one token per program.

Compile-time modes: 0 isotropic, 1 token alpha gates, 2 token beta gate,
3 split steps with shared beta, 4 single step with a parallel-axis correction, 5 channel-wise split steps and beta,
6 scalar split steps and beta with a direct velocity write, 7 independent tangent velocity coefficients,
8 and 9 split the normalized force directly, with tied and independent velocity gains respectively.
"""
import torch
import triton
import triton.language as tl


# Compile-time velocity policies: transport + projection, projection only, or ambient.
_GEOMETRY = {'parallel_transport': 0, 'project_only': 1, 'ambient': 2}


@triton.jit
def _mean(x, D: tl.constexpr):
    if len(x.shape) == 2:
        return tl.sum(x, 1)[:, None] / D
    else:
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
                  D: tl.constexpr, MODE: tl.constexpr, SOFT_FORCE: tl.constexpr):
    force_sq = _mean(force * force, D)
    force_inv = tl.rsqrt(force_sq + FLOOR2) if SOFT_FORCE else tl.rsqrt(tl.maximum(force_sq, FLOOR2))
    target = force * force_inv
    delta = target if MODE == 8 or MODE == 9 else target - h
    if MODE == 1:
        split = _split(delta, v, P, D)
        perp, par, along = split[0], split[1], split[2]
        mixed = (h + perp * delta) + ((par - perp) * along) * v
    elif MODE == 4:
        along = _mean(delta * v, D) / tl.maximum(_mean(v * v, D), 1e-12)
        mixed = h + tl.load(P) * (delta + (tl.load(P + 1) * along) * v)
    elif MODE == 2 or MODE == 3 or MODE == 5 or MODE == 6 or MODE == 7 or MODE == 8 or MODE == 9:
        if MODE == 5:
            channels = tl.arange(0, D)
            perp, par = tl.load(P + channels), tl.load(P + D + channels)
        else:
            perp, par = tl.load(P), tl.load(P + 1)
        along = _mean(delta * v, D) / tl.maximum(_mean(v * v, D), 1e-12)
        v_weight = (par - perp) * along
        if MODE == 6:
            v_weight += tl.load(P + 3)  # rho*v shares the existing velocity multiply.
        mixed = (h + perp * delta) + v_weight * v
    else:
        mixed = h + tl.load(P) * delta
    mixed_sq = _mean(mixed * mixed, D)
    mixed_inv = tl.rsqrt(tl.maximum(mixed_sq, 1e-12))
    new_h = mixed * mixed_inv
    return new_h, mixed_inv, mixed_sq, target, force_inv, force_sq


@triton.jit
def _velocity_axis(n, v, chord, D: tl.constexpr, AMBIENT: tl.constexpr):
    if AMBIENT:
        u, d, vn, dn = v, chord, 0., 0.
    else:
        vn, dn = _mean(v * n, D), _mean(chord * n, D)
        u, d = v - vn * n, chord - dn * n
    u2 = _mean(u * u, D)
    den = tl.maximum(u2, 1e-12)
    along = _mean(u * d, D) / den
    return u, d, vn, dn, u2, den, along


@triton.jit
def _write_values(h, v, force, P, BETA: tl.constexpr, FLOOR2: tl.constexpr,
                  D: tl.constexpr, MODE: tl.constexpr, GEOMETRY: tl.constexpr, SOFT_FORCE: tl.constexpr):
    new_h, mixed_inv, mixed_sq, target, force_inv, force_sq = _position(h, v, force, P, FLOOR2, D, MODE, SOFT_FORCE)
    bisector = h + new_h
    bisector_sq = _mean(bisector * bisector, D)
    denom = tl.maximum(bisector_sq, 1e-12)
    scale = 2.0 * _mean(v * bisector, D) / denom
    carried = v - scale * bisector if GEOMETRY == 0 else v
    if MODE == 2:
        geometry = _history_alignment(new_h, carried, new_h - h, D)
        beta = tl.sigmoid(tl.load(P + 2) + tl.load(P + 3) * geometry[0])
    elif MODE == 5:
        beta = tl.load(P + 2 * D + tl.arange(0, D))
    elif MODE == 1 or MODE == 3 or MODE == 6 or MODE == 7 or MODE == 8 or MODE == 9:
        beta = tl.load(P + 2)
    else:
        beta = BETA
    if MODE == 7 or MODE == 9:
        chord = new_h - h
        along = _velocity_axis(new_h, carried, chord, D, GEOMETRY == 2)[6]
        b, c = tl.load(P + 3), tl.load(P + 4)
        # The tangent policy projects once after mixing; ambient keeps the raw chord.
        velocity = beta * carried + c * chord + ((b - c) * along) * carried
    else:
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
                  NORM: tl.constexpr, MEMORY, GAIN, DMEM, DGAIN, CORRECTION: tl.constexpr):
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
        if CORRECTION:
            memory = tl.load(MEMORY + offsets).to(tl.float32)
            gain = tl.load(GAIN)
            q = (q + gain * memory).to(tl.bfloat16).to(tl.float32)
        inv = tl.rsqrt(_mean(q * q, D) + 1.1920928955078125e-7)
        normalized = q * inv
    if BACKWARD:
        dq = tl.load(Q + offsets).to(tl.float32)
        if NORM:
            dq = ((dq - normalized * _mean(dq * normalized, D)) * inv).to(tl.bfloat16).to(tl.float32)
        if CORRECTION:
            tl.store(DMEM + offsets, gain * dq)
            tl.store(DGAIN + row, tl.sum(dq * memory, 0))
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
                                                  None, None, None, None, False, enable_fp_fusion=False)
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
                              None, None, None, None, False, enable_fp_fusion=False)
        return dh, dv, partials.sum().reshape_as(mu), None, None


def bf16_query(h, v, mu, ambient, normalize=False):
    return _BF16Query.apply(h.contiguous(), v.contiguous(), mu, ambient, normalize)


class _CorrectedQuery(torch.autograd.Function):
    """Fuse BF16 preview, additive memory read and branch pre-norm in one pass."""
    @staticmethod
    def forward(ctx, h, v, mu, memory, gain):
        rows, d = h.numel() // h.size(-1), h.size(-1)
        out = torch.empty_like(h)
        _query_kernel[(rows,)](h, v, mu, out, None, None, None, d, False, False, True,
                               memory, gain, None, None, True, enable_fp_fusion=False)
        ctx.save_for_backward(h, v, mu, memory, gain)
        return out

    @staticmethod
    def backward(ctx, grad):
        h, v, mu, memory, gain = ctx.saved_tensors
        rows, d = h.numel() // h.size(-1), h.size(-1)
        dh, dv, dm = torch.empty_like(h), torch.empty_like(v), torch.empty_like(memory)
        partials = mu.new_empty((2, rows))
        _query_kernel[(rows,)](h, v, mu, grad.contiguous(), dh, dv, partials[0], d, False, True, True,
                               memory, gain, dm, partials[1], True, enable_fp_fusion=False)
        grads = partials.sum(1)
        return dh, dv, grads[:1], dm, grads[1:]


def corrected_query(h, v, mu, memory, gain):
    return _CorrectedQuery.apply(h.contiguous(), v.contiguous(), mu, memory.contiguous(), gain)


@triton.jit
def _forward(H, V, F, P, NH, NV, BETA: tl.constexpr, FLOOR2: tl.constexpr,
             D: tl.constexpr, MODE: tl.constexpr, Q, TERMINAL: tl.constexpr,
             PREVIEW: tl.constexpr, GEOMETRY: tl.constexpr, TOKEN_CONTROLS: tl.constexpr, SOFT_FORCE: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * D + tl.arange(0, D)
    if TOKEN_CONTROLS:
        P += tl.program_id(0).to(tl.int64) * 6 + 1
    h = tl.load(H + offsets).to(tl.float32)
    v = tl.load(V + offsets).to(tl.float32)
    force = tl.load(F + offsets).to(tl.float32)
    if TERMINAL:
        position = _position(h, v, force, P, FLOOR2, D, MODE, SOFT_FORCE)
        tl.store(NH + offsets, position[0])
    else:
        values = _write_values(h, v, force, P, BETA, FLOOR2, D, MODE, GEOMETRY, SOFT_FORCE)
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
              PREVIEW: tl.constexpr, GEOMETRY: tl.constexpr, ROW_TILE: tl.constexpr,
              TOKEN_CONTROLS: tl.constexpr, SOFT_FORCE: tl.constexpr):
    row = tl.program_id(0)
    # Shared controls accumulate one gradient per token for a later reduction.
    # Token controls instead write their own [mu, perp, par, a, b, c] row.
    if TOKEN_CONTROLS:
        P += row.to(tl.int64) * 6 + 1
        dp_row = DP + row.to(tl.int64) * 6 + 1
        dp_stride = 1
        tl.store(dp_row - 1, 0.)  # mu enters the preview only
        if TERMINAL:
            tl.store(dp_row + 2, 0.)
            tl.store(dp_row + 3, 0.)
            tl.store(dp_row + 4, 0.)
    else:
        dp_row = DP + row
        dp_stride = ROWS
    if MODE == 5:
        # Reduce coefficient gradients over a small tile of tokens before writing
        # them out. No atomics or full [tokens, parameters] gradient buffer.
        channels = tl.arange(0, D)
        token = row * ROW_TILE + tl.arange(0, ROW_TILE)
        offsets = token[:, None].to(tl.int64) * D + channels[None, :]
        live = token[:, None] < ROWS
    else:
        offsets = row.to(tl.int64) * D + tl.arange(0, D)
        live = True
    h = tl.load(H + offsets, mask=live, other=0.).to(tl.float32)
    v = tl.load(V + offsets, mask=live, other=0.).to(tl.float32)
    force = tl.load(F + offsets, mask=live, other=0.).to(tl.float32)
    dh_next = tl.load(DNH + offsets, mask=live, other=0.).to(tl.float32)
    if TERMINAL:
        new_h, mixed_inv, mixed_sq, target, force_inv, force_sq = _position(h, v, force, P, FLOOR2, D, MODE, SOFT_FORCE)
        d_new_h = dh_next
        d_h = tl.full((D,), 0., tl.float32)
        d_v = tl.full((D,), 0., tl.float32)
    else:
        (new_h, velocity, radial, bisector, scale, denom, bisector_sq, mixed_inv,
         mixed_sq, target, force_inv, force_sq, carried, beta) = _write_values(h, v, force, P, BETA, FLOOR2, D, MODE, GEOMETRY, SOFT_FORCE)
        # Reverse the tangent projection, then the short-arc parallel transport.
        upstream = tl.load(DNV + offsets, mask=live, other=0.).to(tl.float32)
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
        if MODE == 7 or MODE == 9:
            chord = new_h - h
            u, d, vn, dn, u2, den, along = _velocity_axis(new_h, carried, chord, D, GEOMETRY == 2)
            b, c = tl.load(P + 3), tl.load(P + 4)
            du_dot = _mean(d_velocity * carried, D)
            gb = du_dot * along
            tl.store(dp_row + 2 * dp_stride, D * du_dot)
            tl.store(dp_row + 3 * dp_stride, D * gb)
            tl.store(dp_row + 4 * dp_stride, D * (_mean(d_velocity * chord, D) - gb))
            # Differentiate the moving axis, both tangent projections and the quotient.
            z = (b - c) * du_dot / den
            du = z * (d - tl.where(u2 >= 1e-12, 2. * along * u, 0.))
            dd = z * u
            d_carried = (beta + (b - c) * along) * d_velocity + du
            d_chord = c * d_velocity + dd
            if GEOMETRY == 2:
                d_new_h = dh_next + d_chord
            else:
                un, cn = _mean(du * new_h, D), _mean(dd * new_h, D)
                d_carried -= un * new_h
                d_chord -= cn * new_h
                d_new_h = dh_next + d_chord - radial * upstream - dot * velocity
                d_new_h -= vn * du + un * carried + dn * dd + cn * chord
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
    delta = target if MODE == 8 or MODE == 9 else target - h
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
    elif MODE == 5:
        perp = tl.load(P + channels)
        gain = tl.load(P + D + channels) - perp
        v2 = _mean(v * v, D)
        den = tl.maximum(v2, 1e-12)
        along = _mean(delta * v, D) / den
        # The coefficient is a vector and must be INSIDE this contraction.
        d_dot = _mean(d_mixed * gain * v, D) / den
        d_delta = perp * d_mixed + d_dot * v
        d_v += (gain * along) * d_mixed + d_dot * (delta - tl.where(v2 >= 1e-12, 2. * along * v, 0.))
        d_h += d_mixed - d_delta
        par_dir = along * v
        tile_offset = row * D + channels
        stride = tl.num_programs(0) * D
        tl.store(DP + tile_offset, tl.sum(tl.where(live, d_mixed * (delta - par_dir), 0.), 0))
        tl.store(DP + stride + tile_offset, tl.sum(tl.where(live, d_mixed * par_dir, 0.), 0))
        tl.store(DP + 2 * stride + tile_offset,
                 tl.sum(tl.where(live, d_velocity * (carried - (new_h - h)), 0.), 0))
    elif MODE == 2 or MODE == 3 or MODE == 4 or MODE == 6 or MODE == 7 or MODE == 8 or MODE == 9:
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
        v_weight = gain * along
        if MODE == 6:
            v_weight += tl.load(P + 3)
            tl.store(DP + 3 * ROWS + row, D * dot_v)
        d_v += v_weight * d_mixed + d_dot * (delta - tl.where(v2 >= 1e-12, 2. * along * v, 0.))
        # Only a target displacement has a direct -h path through delta.
        d_h += d_mixed if MODE == 8 or MODE == 9 else d_mixed - d_delta
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
        tl.store(dp_row, mixed_inv * tl.sum(d_new_h * base_dir, 0))
        tl.store(dp_row + dp_stride, mixed_inv * tl.sum(d_new_h * axis_dir, 0))
        if (MODE == 3 or MODE == 6 or MODE == 8) and not TERMINAL:
            tl.store(DP + 2 * ROWS + row, tl.sum(d_velocity * (carried - (new_h - h)), 0))
    else:
        alpha = tl.load(P)
        d_h += (1. - alpha) * d_mixed
        d_delta = alpha * d_mixed
        tl.store(DP + row, tl.sum(d_mixed * delta, 0))
    dot_target = _mean(d_delta * target, D)
    # The soft denominator has a norm derivative at every nonzero force magnitude.
    if SOFT_FORCE:
        d_force = force_inv * (d_delta - target * dot_target)
    else:
        d_force = force_inv * (d_delta - tl.where(force_sq >= FLOOR2, target * dot_target, 0.))
    tl.store(DH + offsets, d_h, mask=live)
    tl.store(DV + offsets, d_v, mask=live)
    tl.store(DF + offsets, d_force, mask=live)


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
    def forward(ctx, h, v, force, params, fixed_beta, floor, mode, geometry, soft_tau=0., *projection):
        width = h.size(-1)
        assert h.is_cuda and h.dtype == v.dtype and h.dtype in (torch.bfloat16, torch.float32)
        assert params.dtype == torch.float32
        assert h.shape == v.shape == force.shape and params.numel() == (1, 5, 4, 3, 2, 3 * width, 4, 5, 3, 5)[mode]
        assert width == triton.next_power_of_2(width) and force.dtype in (torch.bfloat16, torch.float32)
        new_h, new_v = torch.empty_like(h), torch.empty_like(v)
        rows = h.numel() // width
        denom2 = (soft_tau or floor)**2
        ctx.soft_force = bool(soft_tau)
        _forward[(rows,)](h, v, force, params, new_h, new_v, fixed_beta, denom2,
                          width, mode, params, False, False, geometry, False, ctx.soft_force, num_warps=4, enable_fp_fusion=False)
        ctx.remat = len(projection) == 2
        if ctx.remat:
            features, weight = projection
            ctx.save_for_backward(h, v, params, features, weight)
        else:
            assert len(projection) == 0
            ctx.save_for_backward(h, v, params, force)
        ctx.options = fixed_beta, denom2, width, mode
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
        channel_mode = ctx.options[-1] == 5
        row_tile = 16 if channel_mode else 1
        tiles = triton.cdiv(rows, row_tile)
        shape = (3, tiles, h.size(-1)) if channel_mode else (params.numel(), rows)
        partials = params.new_empty(shape)
        _backward[(tiles,)](h, v, force, params, d_new_h.contiguous(), d_new_v.contiguous(),
                           dh, dv, df, partials, rows, *ctx.options, params, False, False, ctx.geometry, row_tile, False, ctx.soft_force,
                           num_warps=8 if channel_mode else 4, enable_fp_fusion=False)
        return (dh, dv, df, partials.sum(1).reshape_as(params)) + (None,) * (len(ctx.needs_input_grad) - 4)


class TokenWrite(torch.autograd.Function):
    """Use the same native three-scalar write with a separate control row per token."""
    @staticmethod
    def forward(ctx, h, v, force, controls, floor, terminal, soft_tau=0., ambient=False, transport=False):
        h, v, force = h.contiguous(), v.contiguous(), force.contiguous()
        width = h.size(-1)
        rows = h.numel() // width
        nh = torch.empty_like(h)
        nv = None if terminal else torch.empty_like(v)
        ctx.soft_force = bool(soft_tau)
        geometry = 'ambient' if ambient else 'project_only'
        ctx.geometry = _GEOMETRY['parallel_transport' if transport else geometry]
        ctx.mode = 9 if soft_tau else 7
        ctx.floor2 = soft_tau**2 if soft_tau else floor**2
        _forward[(rows,)](h, v, force, controls, nh, nv, 0., ctx.floor2, width, ctx.mode,
                          controls, terminal, False, ctx.geometry, TOKEN_CONTROLS=True, SOFT_FORCE=ctx.soft_force,
                          num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(h, v, force, controls)
        ctx.terminal = terminal
        return nh, nv

    @staticmethod
    def backward(ctx, dnh, dnv):
        h, v, force, controls = ctx.saved_tensors
        width = h.size(-1)
        rows = h.numel() // width
        dh, dv, df = torch.empty_like(h), torch.empty_like(v), torch.empty_like(force)
        dc = torch.empty_like(controls)
        _backward[(rows,)](h, v, force, controls, dnh.contiguous(), None if ctx.terminal else dnv.contiguous(),
                           dh, dv, df, dc, rows, 0., ctx.floor2, width, ctx.mode, controls, ctx.terminal,
                           False, ctx.geometry, 1, TOKEN_CONTROLS=True, SOFT_FORCE=ctx.soft_force, num_warps=4, enable_fp_fusion=False)
        return (dh, dv, df, dc) + (None,) * (len(ctx.needs_input_grad) - 4)


class _SplitBoundary(torch.autograd.Function):
    """Two parameters for a terminal write, or [perp, parallel, beta, next_mu] for a preview."""
    @staticmethod
    def forward(ctx, h, v, force, params, floor, terminal, geometry, soft_tau):
        width = h.size(-1)
        nh = torch.empty_like(h)
        nv = params if terminal else torch.empty_like(v)
        q = params if terminal else torch.empty_like(h, dtype=torch.bfloat16)
        ctx.floor2 = (soft_tau or floor)**2
        ctx.soft_force = bool(soft_tau)
        _forward[(h.numel() // width,)](h, v, force, params, nh, nv, 0., ctx.floor2, width, 3,
                                        q, terminal, not terminal, geometry, False, ctx.soft_force, num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(h, v, force, params)
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
                           0., ctx.floor2, width, 3, dq, terminal, not terminal, ctx.geometry, 1, False, ctx.soft_force,
                           num_warps=4, enable_fp_fusion=False)
        return dh, dv, df, partials.sum(1), None, None, None, None


def split_boundary(site, h, v, force, next_mu):
    if site.update_based:
        raise ValueError('Update-based writes use fused_split_write')
    params = [site.alpha(), site.step_size(site.raw_parallel)]
    if next_mu is not None:
        params.extend((site.beta(), next_mu[0]))
    return _SplitBoundary.apply(h.contiguous(), v.contiguous(), force.contiguous(), torch.stack(params),
                                site.rms_floor, next_mu is None, _GEOMETRY[site.velocity_geometry], site.soft_tau)


def fused_gated_write(h, v, force, params, floor):
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), params, 0., floor, 1, 0)


def fused_beta_write(h, v, force, params, floor):
    """Positive split steps with per-token velocity decay. P holds both steps, bias and slope."""
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), params, 0., floor, 2, 0)


def fused_write(h, v, force, alpha, beta, floor):
    """Isotropic Chord write."""
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), alpha, beta, floor, 0, 0)


def fused_split_write(h, v, force, params, floor, geometry='parallel_transport', projection=(), *, update_based=False, soft_tau=0.):
    """Pack the position steps and velocity coefficients for this write policy."""
    mode = 5 if params.ndim == 2 else 3
    if params.shape == (5,):
        assert geometry == 'project_only'
        mode = 9 if update_based else 7
    elif update_based:
        assert params.shape == (3,)
        mode = 8
    elif params.shape == (4,):
        mode = 6
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), params, 0., floor, mode, _GEOMETRY[geometry], soft_tau, *projection)


def fused_axis_write(h, v, force, params, beta, floor):
    """P holds the isotropic step and kappa; beta is fixed."""
    return _ChordWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), params, beta, floor, 4, 0)
