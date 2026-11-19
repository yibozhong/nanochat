"""Experimental fused spherical Arc write. The branch read stays in the compiled block."""
import torch
import triton
import triton.language as tl


@triton.jit
def _write_values(h, v, force, alpha, BETA: tl.constexpr, FLOOR2: tl.constexpr,
                  WIDTH: tl.constexpr):
    # Each program owns an entire (power-of-two) hidden row.
    force_sq = tl.sum(force * force, 0) / WIDTH
    force_inv = tl.rsqrt(tl.maximum(force_sq, FLOOR2))
    target = force * force_inv
    mixed = h + alpha * (target - h)
    mixed_sq = tl.sum(mixed * mixed, 0) / WIDTH
    mixed_inv = tl.rsqrt(tl.maximum(mixed_sq, 1e-12))
    new_h = mixed * mixed_inv
    bisector = h + new_h
    bisector_sq = tl.sum(bisector * bisector, 0) / WIDTH
    denom = tl.maximum(bisector_sq, 1e-12)
    scale = 2.0 * (tl.sum(v * bisector, 0) / WIDTH) / denom
    carried = v - scale * bisector
    velocity = BETA * carried + (1.0 - BETA) * (new_h - h)
    radial = tl.sum(new_h * velocity, 0) / WIDTH
    return (new_h, velocity, radial, bisector, scale, denom, bisector_sq,
            mixed_inv, mixed_sq, target, force_inv, force_sq)


@triton.jit
def _forward(H, V, F, A, NH, NV, BETA: tl.constexpr, FLOOR2: tl.constexpr, WIDTH: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * WIDTH + tl.arange(0, WIDTH)
    h = tl.load(H + offsets)
    v = tl.load(V + offsets)
    force = tl.load(F + offsets).to(tl.float32)
    values = _write_values(h, v, force, tl.load(A), BETA, FLOOR2, WIDTH)
    new_h, velocity, radial = values[0], values[1], values[2]
    tl.store(NH + offsets, new_h)
    tl.store(NV + offsets, velocity - radial * new_h)


@triton.jit
def _backward(H, V, F, A, DNH, DNV, DH, DV, DF, DA, BETA: tl.constexpr,
              FLOOR2: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    offsets = row.to(tl.int64) * WIDTH + tl.arange(0, WIDTH)
    h = tl.load(H + offsets)
    v = tl.load(V + offsets)
    force = tl.load(F + offsets).to(tl.float32)
    alpha = tl.load(A)
    (new_h, velocity, radial, bisector, scale, denom, bisector_sq, mixed_inv,
     mixed_sq, target, force_inv, force_sq) = _write_values(h, v, force, alpha, BETA, FLOOR2, WIDTH)
    d_new_h = tl.load(DNH + offsets)
    # Reverse the tangent projection, then the Householder transport.
    upstream = tl.load(DNV + offsets)
    dot = tl.sum(upstream * new_h, 0) / WIDTH
    d_velocity = upstream - dot * new_h
    d_new_h += (1.0 - BETA) * d_velocity - radial * upstream - dot * velocity
    d_carried = BETA * d_velocity
    coeff = -2.0 * (tl.sum(d_carried * bisector, 0) / WIDTH) / denom
    d_v = d_carried + coeff * bisector
    d_bisector = coeff * (v - tl.where(bisector_sq >= 1e-12, scale * bisector, 0.0)) - scale * d_carried
    d_h = d_bisector - (1.0 - BETA) * d_velocity
    d_new_h += d_bisector

    # Reverse both clamped normalizations. At the floor, clamp_min passes the gradient.
    dot_h = tl.sum(d_new_h * new_h, 0) / WIDTH
    d_mixed = mixed_inv * (d_new_h - tl.where(mixed_sq >= 1e-12, new_h * dot_h, 0.0))
    d_h += (1.0 - alpha) * d_mixed
    dot_target = tl.sum(d_mixed * target, 0) / WIDTH
    d_force = alpha * force_inv * (d_mixed - tl.where(force_sq >= FLOOR2, target * dot_target, 0.0))
    tl.store(DH + offsets, d_h)
    tl.store(DV + offsets, d_v)
    tl.store(DF + offsets, d_force)
    tl.store(DA + row, tl.sum(d_mixed * (target - h), 0))


class _ArcWrite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, v, force, alpha, beta, floor):
        width = h.size(-1)
        assert h.is_cuda and h.dtype == v.dtype == alpha.dtype == torch.float32
        assert h.shape == v.shape == force.shape and alpha.numel() == 1
        assert width in (128, 512) and force.dtype in (torch.bfloat16, torch.float32)
        new_h, new_v = torch.empty_like(h), torch.empty_like(v)
        rows = h.numel() // width
        _forward[(rows,)](h, v, force, alpha, new_h, new_v, beta, floor**2, width,
                          num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(h, v, force, alpha)
        ctx.beta, ctx.floor = beta, floor
        return new_h, new_v

    @staticmethod
    def backward(ctx, d_new_h, d_new_v):
        h, v, force, alpha = ctx.saved_tensors
        width = h.size(-1)
        rows = h.numel() // width
        d_h, d_v, d_force = torch.empty_like(h), torch.empty_like(v), torch.empty_like(force)
        d_alpha = h.new_empty((rows,))
        _backward[(rows,)](h, v, force, alpha,
                           d_new_h.contiguous(), d_new_v.contiguous(), d_h, d_v, d_force, d_alpha,
                           ctx.beta, ctx.floor**2, width,
                           num_warps=4, enable_fp_fusion=False)
        return d_h, d_v, d_force, d_alpha.sum().reshape_as(alpha), None, None


def fused_write(h, v, force, alpha, beta, floor):
    return _ArcWrite.apply(h.contiguous(), v.contiguous(), force.contiguous(), alpha, beta, floor)


def compile_arc_step(block):
    if block.attn_residual.kappa is not None or block.mlp_residual.kappa is not None:
        raise ValueError('The benchmark kernel implements the original isotropic Arc write')
    def update(site, h, v, force):
        return fused_write(h, v, force, site.alpha(), site.decay, site.rms_floor)

    def step(h, v, cos, sin, cache=None):
        attention = block.attn_residual
        query = attention.query(h, v).to(cos.dtype)
        h, v = update(attention, h, v, block.attention_branch(query, cos, sin, cache))
        ffn = block.mlp_residual
        query = ffn.query(h, v).to(cos.dtype)
        return update(ffn, h, v, block.mlp_branch(query))

    return torch.compile(step, fullgraph=True, dynamic=False, options={'emulate_precision_casts': True})
