from pathlib import Path
root=Path(__file__).resolve().parents[3]
s=(root/'research/connections/triton-moments/upstream/fla/ops/attn/parallel.py').read_text()
start=s.index('@triton.heuristics');end=s.index('@dispatch\ndef parallel_attn_fwd')
kernels=s[start:end]
header=s[:s.index('import torch')]+'''# Local specialization/adaptation; provenance and MIT license in
# research/connections/triton-moments/upstream. Optional FLA features are
# retained in the vendored kernels but the local wrapper accepts causal MHA only.
import torch
import triton
import triton.language as tl

@triton.jit
def exp2(x): return tl.math.exp2(x.to(tl.float32))
@triton.jit
def log2(x): return tl.log2(x.to(tl.float32))
@triton.jit
def unflatten_program_id(X: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    return pid % X, pid // X

'''
(root/'nanochat/fla_attention_kernels.py').write_text(header+kernels)
# Extend the upstream fwd/dq/dkv structure, keeping the same online softmax and loops.
fwd=kernels[:kernels.index('@triton.jit\ndef parallel_attn_bwd_kernel_preprocess')]
bwd=kernels[kernels.index('@triton.heuristics',kernels.index('def parallel_attn_bwd_kernel_preprocess')):]
fwd=fwd.replace('parallel_attn_fwd_kernel','moment_fwd_kernel').replace('    o,\n','    o,\n    stat,\n    raw,\n    MODE: tl.constexpr,\n')
fwd=fwd.replace('    b_m = tl.full', '    b_z = tl.zeros([BT, BV], dtype=tl.float32)\n\n    b_m = tl.full',1)
fwd=fwd.replace('        b_mp = b_m','        b_z = b_z * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), phi(b_v, MODE).to(b_v.dtype))\n        b_mp = b_m')
fwd=fwd.replace('    b_m += log2(b_acc)', '''    b_z = b_z / b_acc[:, None]
    # Center with the same BF16 phi evaluation as the V side. Keep the
    # accumulators on chip until centering; no BF16 raw-moment subtraction.
    b_mean = b_o.to(o.dtype.element_ty)
    if MODE == 1:
        b_stat = b_z
    else:
        b_stat = b_z - phi(b_mean, MODE).to(b_mean.dtype).to(tl.float32)
        if MODE == 2:
            b_stat = tl.maximum(b_stat, 0.)
    p_stat = stat + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    p_raw = raw + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    tl.store(p_stat, b_stat, mask=m_q[:, None] & (o_v[None, :] < V))
    tl.store(p_raw, b_z, mask=m_q[:, None] & (o_v[None, :] < V))
    b_m += log2(b_acc)''')
bwd=bwd.replace('parallel_attn_bwd_kernel_', 'moment_bwd_kernel_').replace('    do,\n','    do,\n    dz,\n    MODE: tl.constexpr,\n')
# q-centric and k-centric both load the extra upstream derivative with same strides.
needle='    b_do = tl.load(p_do, mask=m_q[:, None] & (o_v[None, :] < V), other=0.0).to(v.dtype.element_ty)'
bwd=bwd.replace(needle,needle+'\n    b_dz = tl.load(dz + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :], mask=m_q[:, None] & (o_v[None, :] < V), other=0.).to(v.dtype.element_ty)')
# The nested loads have eight spaces and need the same indentation for inserted lines.
bwd=bwd.replace('        b_do = tl.load(p_do, mask=m_q[:, None] & (o_v[None, :] < V), other=0.0).to(v.dtype.element_ty)\n    b_dz', '        b_do = tl.load(p_do, mask=m_q[:, None] & (o_v[None, :] < V), other=0.0).to(v.dtype.element_ty)\n        b_dz')
bwd=bwd.replace('b_dp = tl.dot(b_do, b_v)','b_dp = tl.dot(b_do, b_v) + tl.dot(b_dz, phi(b_v, MODE).to(b_v.dtype))')
bwd=bwd.replace('b_dp = tl.dot(b_v, tl.trans(b_do))','b_dp = tl.dot(b_v, tl.trans(b_do)) + tl.dot(phi(b_v, MODE).to(b_v.dtype), tl.trans(b_dz))')
bwd=bwd.replace('    b_dv = tl.zeros([BT, BV], dtype=tl.float32)','    b_dv = tl.zeros([BT, BV], dtype=tl.float32)\n    b_dzv = tl.zeros([BT, BV], dtype=tl.float32)')
bwd=bwd.replace('        b_dv = tl.dot(b_p.to(b_do.dtype), b_do, b_dv)','        b_dv = tl.dot(b_p.to(b_do.dtype), b_do, b_dv)\n        b_dzv = tl.dot(b_p.to(b_do.dtype), b_dz, b_dzv)')
bwd=bwd.replace('    b_dk = b_dk * scale','    b_dv += b_dzv * phi_grad(b_v, MODE)\n    b_dk = b_dk * scale')
phi='''@triton.jit
def phi(x, MODE: tl.constexpr):
    x = x.to(tl.float32)
    if MODE == 3:
        return x * tl.sigmoid(x)
    if MODE == 4:
        return tl.abs(x)
    return x * x

@triton.jit
def phi_grad(x, MODE: tl.constexpr):
    x = x.to(tl.float32)
    if MODE == 3:
        s = tl.sigmoid(x)
        return s * (1. + x * (1. - s))
    if MODE == 4:
        # Match torch.abs: the chosen subgradient at zero is zero.
        return tl.where(x > 0., 1., tl.where(x < 0., -1., 0.))
    return 2. * x

@triton.jit
def moment_preprocess(O, R, S, DO, DS, DM, DZ, Delta, D: tl.constexpr, BD: tl.constexpr, MODE: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BD)
    mu = tl.load(O + row*D+d, d<D, 0).to(tl.float32)
    raw = tl.load(R + row*D+d, d<D, 0).to(tl.float32)
    gs = tl.load(DS + row*D+d, d<D, 0).to(tl.float32)
    if MODE == 2:
        stat = tl.load(S + row*D+d, d<D, 0)
        gs = tl.where(stat > 0, gs, 0.)
    gm = tl.load(DO + row*D+d, d<D, 0).to(tl.float32)
    if MODE != 1:
        gm -= gs * phi_grad(mu, MODE)
    # Dot-product inputs have the original activation dtype.
    gm = gm.to(DM.dtype.element_ty)
    gs = gs.to(DZ.dtype.element_ty)
    tl.store(DM + row*D+d, gm, d<D)
    tl.store(DZ + row*D+d, gs, d<D)
    tl.store(Delta+row, tl.sum(mu*gm.to(tl.float32) + raw*gs.to(tl.float32)))

'''
(root/'nanochat/moment_attention_kernels.py').write_text(header+phi+fwd+bwd)

# Three simultaneous value streams: V, V^2, V^3. Keep the validated two-stream
# kernels above byte-for-byte unchanged; the new variant has its own launches.
third_fwd=fwd.replace('moment_fwd_kernel','third_fwd_kernel')
third_fwd=third_fwd.replace('    stat,\n','    stat,\n    third,\n    raw3,\n')
third_fwd=third_fwd.replace('    b_z = tl.zeros([BT, BV], dtype=tl.float32)',
    '    b_z = tl.zeros([BT, BV], dtype=tl.float32)\n    b_z3 = tl.zeros([BT, BV], dtype=tl.float32)')
needle='        b_z = b_z * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), phi(b_v, MODE).to(b_v.dtype))'
assert third_fwd.count(needle)==2
third_fwd=third_fwd.replace(needle,needle+'\n        b_z3 = b_z3 * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), cube(b_v).to(b_v.dtype))')
third_fwd=third_fwd.replace('    b_m += log2(b_acc)', '''    b_z3 = b_z3 / b_acc[:, None]
    mu = b_mean.to(tl.float32)
    # Algebraically m3 - 3*mu*m2 + 2*mu^3. Center with the SAME
    # BF16 transformed values on both sides; singleton gaps then cancel.
    mu2 = phi(b_mean, MODE).to(b_mean.dtype).to(tl.float32)
    mu3 = cube(b_mean).to(b_mean.dtype).to(tl.float32)
    b_third = (b_z3 - mu3) - 3. * mu * (b_z - mu2)
    offs = (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    tl.store(third + offs, b_third, mask=m_q[:, None] & (o_v[None, :] < V))
    tl.store(raw3 + offs, b_z3, mask=m_q[:, None] & (o_v[None, :] < V))
    b_m += log2(b_acc)''')
third_bwd=bwd.replace('moment_bwd_kernel_', 'third_bwd_kernel_').replace('    dz,\n','    dz,\n    dz3,\n')
needle='b_dz = tl.load(dz + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :], mask=m_q[:, None] & (o_v[None, :] < V), other=0.).to(v.dtype.element_ty)'
lines=[]
for line in third_bwd.splitlines():
    lines.append(line)
    if line.strip()==needle:
        lines.append(line.replace('b_dz =','b_dz3 =').replace('tl.load(dz +','tl.load(dz3 +'))
third_bwd='\n'.join(lines)+'\n'
third_bwd=third_bwd.replace('tl.dot(b_dz, phi(b_v, MODE).to(b_v.dtype))',
    'tl.dot(b_dz, phi(b_v, MODE).to(b_v.dtype)) + tl.dot(b_dz3, cube(b_v).to(b_v.dtype))')
third_bwd=third_bwd.replace('tl.dot(phi(b_v, MODE).to(b_v.dtype), tl.trans(b_dz))',
    'tl.dot(phi(b_v, MODE).to(b_v.dtype), tl.trans(b_dz)) + tl.dot(cube(b_v).to(b_v.dtype), tl.trans(b_dz3))')
third_bwd=third_bwd.replace('    b_dzv = tl.zeros([BT, BV], dtype=tl.float32)',
    '    b_dzv = tl.zeros([BT, BV], dtype=tl.float32)\n    b_dzv3 = tl.zeros([BT, BV], dtype=tl.float32)')
needle='        b_dzv = tl.dot(b_p.to(b_do.dtype), b_dz, b_dzv)'
assert third_bwd.count(needle)==2
third_bwd=third_bwd.replace(needle,needle+'\n        b_dzv3 = tl.dot(b_p.to(b_do.dtype), b_dz3, b_dzv3)')
third_bwd=third_bwd.replace('    b_dv += b_dzv * phi_grad(b_v, MODE)',
    '    b_dv += b_dzv * phi_grad(b_v, MODE) + b_dzv3 * (3. * b_v.to(tl.float32) * b_v.to(tl.float32))')
third_helpers=phi[:phi.index('@triton.jit\ndef moment_preprocess')]+'''@triton.jit
def cube(x):
    x = x.to(tl.float32)
    return x * x * x

@triton.jit
def third_preprocess(O, R2, R3, S, DO, DS, DT, DM, D2, D3, Delta, D: tl.constexpr, BD: tl.constexpr, T: tl.constexpr, H: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BD)
    mu = tl.load(O + row*D+d, d<D, 0).to(tl.float32)
    r2 = tl.load(R2 + row*D+d, d<D, 0).to(tl.float32)
    r3 = tl.load(R3 + row*D+d, d<D, 0).to(tl.float32)
    gs = tl.load(DS + row*D+d, d<D, 0).to(tl.float32)
    stat = tl.load(S + row*D+d, d<D, 0)
    gs = tl.where(stat > 0, gs, 0.)
    gt = tl.load(DT + row*D+d, d<D, 0).to(tl.float32)
    # A singleton distribution has identically zero third central moment.
    gt = tl.where((row // H) % T == 0, 0., gt)
    gm = tl.load(DO + row*D+d, d<D, 0).to(tl.float32)
    mu2 = (mu * mu).to(O.dtype.element_ty).to(tl.float32)
    gm = gm - 2. * mu * gs + (3. * mu * mu + 3. * mu2 - 3. * r2) * gt
    g2 = gs - 3. * mu * gt
    gm = gm.to(DM.dtype.element_ty)
    g2 = g2.to(D2.dtype.element_ty)
    gt = gt.to(D3.dtype.element_ty)
    tl.store(DM + row*D+d, gm, d<D)
    tl.store(D2 + row*D+d, g2, d<D)
    tl.store(D3 + row*D+d, gt, d<D)
    tl.store(Delta+row, tl.sum(mu*gm.to(tl.float32) + r2*g2.to(tl.float32) + r3*gt.to(tl.float32)))

'''
(root/'nanochat/third_moment_attention_kernels.py').write_text(header+third_helpers+third_fwd+third_bwd)
