from copy import deepcopy

import pytest
import torch

from nanochat.connection_model import ConnectionConfig
from nanochat.lookahead import LookaheadResidual
from nanochat.momentum import normalize_residual, tangent_project
from test_momentum import gpu


@pytest.mark.parametrize('mode', ['lookahead_transport', 'lookahead_axis_write', 'lookahead_split_write',
                                  'lookahead_split_learned_beta'])
@pytest.mark.parametrize('width,dtype', [(128,torch.float32),(512,torch.bfloat16)])
def test_shared_chord_write_vjp(mode, width, dtype):
    cfg = ConnectionConfig(streams=1, residual_dynamics='ngpt_lookahead', ngpt_velocity_mode=mode,
                           ngpt_force_rms_floor=1e-6)
    site = LookaheadResidual(cfg).cuda()
    with torch.no_grad():
        if site.kappa is not None:
            site.kappa.fill_(-1.3)
        if site.raw_parallel is not None:
            site.raw_parallel.add_(.8)
        if site.raw_beta is not None:
            site.raw_beta.fill_(.6)
    h = normalize_residual(torch.randn(37,width,device='cuda'))
    v = tangent_project(h,torch.randn_like(h))
    f = torch.randn_like(h)
    v[0].zero_()
    f[1].zero_()
    f[2].mul_(1e-8)
    h[3].fill_(1.)
    f[3].fill_(1.)
    v[3] = tangent_project(h[3], v[3])
    inputs = (h.requires_grad_(), v.requires_grad_(), f.to(dtype).requires_grad_())
    oracle = deepcopy(site).double()
    precise = tuple(t.detach().double().requires_grad_() for t in inputs)
    expected = oracle.reference_write(*precise)
    actual = site(*inputs)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a.double(),b,rtol=3e-5,atol=3e-5)
    up = tuple(torch.randn_like(t) for t in actual)
    names = [n for n,_ in site.named_parameters() if n != 'mu']
    args = (*inputs, *(site.get_parameter(n) for n in names))
    ref_args = (*precise, *(oracle.get_parameter(n) for n in names))
    grads = torch.autograd.grad(actual,args,up)
    refs = torch.autograd.grad(expected,ref_args,tuple(t.double() for t in up))
    for i,(grad,ref) in enumerate(zip(grads,refs)):
        groups = [(grad[j],ref[j]) for j in range(4)] + [(grad[4:],ref[4:])] if i < 3 else [(grad,ref)]
        for g,r in groups:
            err = (g.double()-r).norm()
            assert err <= (.008 if g.dtype==torch.bfloat16 else .0005)*r.norm()+1e-5, (mode,i,err,r.norm())
