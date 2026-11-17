"""GPU-only residual, inference and optimizer regression checks."""
from dataclasses import asdict, replace

import pytest
import torch

from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for
from nanochat.engine import KVCache
from nanochat.packed_training import seed_all, snapshot, restore_rng
from nanochat.connection_experiment import configure_attention
from nanochat.momentum import approximate_momentum_step
from test_connection_training_resume import equal_tree
@pytest.fixture(autouse=True)
def gpu():
    assert torch.cuda.is_available(), 'Run via the bounded Condor GPU debug job'
    configure_attention('fa2')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    seed_all(42)


def inputs():
    x = torch.randint(0, 256, (2, 16), device='cuda')
    return x, (x + 3) % 256


@pytest.fixture(params=[('heavy_ball', 'token'), ('heavy_ball', 'zero'),
                        ('an_heavy_ball', 'token'), ('an_heavy_ball', 'zero')])
def setting(request):
    mode, init = request.param
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32,
        vocab_size=256, streams=1, residual_dynamics=mode, velocity_init=init)


def make(c, compiled=False, checkpoint=False, perturb=True):
    model = build_connection_model(replace(c, activation_checkpoint=checkpoint)).prepare(compile_blocks=compiled)
    if perturb:
        with torch.no_grad():
            for b in model.blocks:
                b.attn.c_proj.weight.normal_(std=.02)
                b.mlp.c_proj.weight.normal_(std=.02)
    return model


def test_backbone_weights_and_optimizer_membership(setting):
    model = make(setting, perturb=False)
    seed_all(42)
    base = make(replace(setting, residual_dynamics='vanilla', velocity_init='zero'), perturb=False)
    for name, value in base.state_dict().items():
        assert torch.equal(model.state_dict()[name], value), name
    opt, _ = optimizer_for(model)
    if model.velocity_embedding is not None:
        group = next(g for g in opt.param_groups if any(p is model.velocity_embedding.weight for p in g['params']))
        assert group['kind'] == 'adamw' and group['weight_decay'] == .001
        assert any(p is model.wte.weight for p in group['params'])
        assert model.velocity_embedding.weight.data_ptr() != model.wte.weight.data_ptr()
        assert .018 < model.velocity_embedding.weight.float().std() < .022
    assert ConnectionConfig.from_checkpoint(asdict(setting)) == setting


def test_initial_state_and_independent_update(setting):
    model = make(setting, perturb=False)
    x, _ = inputs()
    h, v = model.initial_state(model.wte(x), x)
    if setting.velocity_init == 'token':
        torch.testing.assert_close(v, model.velocity_embedding(x).float(), rtol=0, atol=0)
    else:
        assert model.velocity_embedding is None and torch.count_nonzero(v) == 0
    force = torch.randn_like(h)
    update = model.blocks[0].attn_residual
    beta, gamma, eta = update.coefficients()
    if setting.residual_mode == 'heavy_ball':
        actual, carried = update(h, v, force)
        u = beta*v + gamma*force
        expected = h + eta*u
    else:
        moments = model.initial_moments(h)
        actual, carried, next_moments = update(h, v, moments, force)
        z = force / (force.square().mean(-1, keepdim=True) + setting.an_force_rms_floor**2).sqrt()
        u = beta*v + gamma*z
        qu = beta**2*moments[0] + gamma**2
        scale = (1 + 2*eta*beta*moments[1] + eta**2*qu).sqrt()
        expected = (h + eta*u)/scale
        torch.testing.assert_close(next_moments[0], qu)
        torch.testing.assert_close(next_moments[1], (beta*moments[1] + eta*qu)/scale)
    torch.testing.assert_close(carried, u, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('checkpoint', [False, True])
def test_compiled_checkpoint_forward_backward(setting, checkpoint):
    eager = make(setting)
    actual = make(setting, compiled=True, checkpoint=checkpoint)
    actual.load_state_dict(eager.state_dict())
    x, y = inputs()
    a = eager(x, y, dense=True); b = actual(x, y, dense=True)
    a.backward(); b.backward()
    torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4)
    for (name, p), (_, q) in zip(eager.named_parameters(), actual.named_parameters()):
        assert p.grad is not None and q.grad is not None
        assert torch.isfinite(q.grad).all(), name
        error = (p.grad.float()-q.grad.float()).norm()/p.grad.float().norm().clamp_min(1e-10)
        assert error < .04, (name, error.item())


def test_cce_gradient_and_velocity_learning(setting):
    model = make(setting, compiled=True); x, y = inputs()
    loss = model(x, y, dense=True); loss.backward()
    expected = {n:p.grad.clone() for n,p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    other = model(x, y); other.backward()
    torch.testing.assert_close(loss, other, rtol=1e-4, atol=1e-4)
    for n, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        error = (p.grad.float()-expected[n].float()).norm()/expected[n].float().norm().clamp_min(1e-10)
        assert error < .035, (n, error.item())
    if model.velocity_embedding is not None:
        assert model.velocity_embedding.weight.grad.abs().sum() > 0
        assert model.blocks[0].attn_residual.raw.grad[0].abs() > 0
        untouched = torch.ones(256, device='cuda', dtype=torch.bool); untouched[x.unique()] = False
        assert torch.count_nonzero(model.velocity_embedding.weight.grad[untouched]) == 0
    elif setting.residual_mode == 'heavy_ball':
        assert model.blocks[0].attn_residual.raw.grad[0] == 0
        assert model.blocks[-1].mlp_residual.raw.grad.abs().sum() > 0


@torch.no_grad()
def test_causal_batch_and_cached_chunks(setting):
    model = make(setting).eval(); x, _ = inputs()
    full = model(x)
    changed=x.clone(); changed[:,7:] = (changed[:,7:]+19)%256
    assert torch.equal(full[:,:7], model(changed)[:,:7])
    changed=x.clone(); changed[1] = (changed[1]+19)%256
    assert torch.equal(full[0], model(changed)[0])
    cache=KVCache(2,1,32,128,2,'cuda',torch.bfloat16)
    chunks=torch.cat([model(x[:,:5],kv_cache=cache),model(x[:,5:10],kv_cache=cache),model(x[:,10:],kv_cache=cache)],1)
    cache.reset()
    tokens=torch.cat([model(x[:,i:i+1],kv_cache=cache) for i in range(x.shape[1])],1)
    for value in (chunks,tokens):assert (full-value).norm()/full.norm()<.025
    cache.reset()
    torch.testing.assert_close(model(x,kv_cache=cache),full,rtol=0,atol=0)


def test_exact_optimizer_checkpoint_next_update(setting, tmp_path):
    model=make(setting); opt,_=optimizer_for(model); x,y=inputs()
    model(x,y).backward();opt.step();opt.zero_grad(set_to_none=True)
    path=tmp_path/'state.pt';torch.save(snapshot(model,opt,1,config=asdict(setting)),path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    restored=make(ConnectionConfig.from_checkpoint(state['config']))
    other,_=optimizer_for(restored)
    restored.load_state_dict(state['model']);other.load_state_dict(state['optimizer'])
    equal_tree(model.state_dict(),restored.state_dict());equal_tree(opt.state_dict(),other.state_dict())
    restore_rng(state['rng']);expected=torch.rand(4,device='cuda')
    restore_rng(state['rng']);assert torch.equal(expected,torch.rand(4,device='cuda'))
    model(x,y).backward(); gradients={n:p.grad.clone() for n,p in model.named_parameters()}
    for m,o in ((model,opt),(restored,other)):
        o.zero_grad(set_to_none=True)
        for n,p in m.named_parameters():p.grad=gradients[n].clone()
        o.step()
    equal_tree(model.state_dict(),restored.state_dict());equal_tree(opt.state_dict(),other.state_dict())


@torch.no_grad()
def test_hb_zero_force_identity_and_zero_embedding():
    c = ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32,
        vocab_size=256, streams=1, residual_dynamics='heavy_ball', velocity_init='zero')
    model = make(c, perturb=False).eval(); x, _ = inputs()
    h, v = model.initial_state(model.wte(x), x)
    torch.testing.assert_close(model.backbone(model.wte(x), token_ids=x), h, rtol=0, atol=0)
    assert torch.count_nonzero(v) == 0
    base = make(replace(c, residual_dynamics='vanilla'), perturb=False).eval()
    base.load_state_dict({n:t for n,t in model.state_dict().items() if '_residual.' not in n})
    torch.testing.assert_close(model(x), base(x), rtol=0, atol=0)


def test_defaults_and_legacy_checkpoint_retirement():
    assert ConnectionConfig().residual_mode == 'vanilla'
    for mode in ('heavy_ball', 'an_heavy_ball'):
        c = ConnectionConfig(residual_dynamics=mode)
        assert c.velocity_init == 'token'
        assert ConnectionConfig.from_checkpoint(asdict(c)) == c
    c = ConnectionConfig(residual_dynamics='heavy_ball')
    old = asdict(c); old['riemannian_momentum'] = False
    old.pop('an_force_rms_floor'); old.pop('velocity_init')
    assert ConnectionConfig.from_checkpoint(old).velocity_init == 'zero'
    assert ConnectionConfig.from_checkpoint(old).momentum_eta_init == 1.
    for fields in ({'riemannian_momentum': True}, {'residual_dynamics': 'sphere'},
                   {'residual_dynamics': 'riemannian'}):
        with pytest.raises(ValueError, match='retired sphere'):
            ConnectionConfig.from_checkpoint(fields)
    for fields in ({'residual_dynamics':'sphere'}, {'an_force_rms_floor':0},
                   {'momentum_beta_init':1}, {'an_force_rms_floor':float('nan')}):
        with pytest.raises(ValueError): ConnectionConfig(**fields)


@pytest.mark.parametrize('zero', [False, True])
def test_an_first_and_second_derivatives(zero):
    h = torch.randn(1,4,device='cuda',dtype=torch.float64,requires_grad=True)
    v = torch.randn_like(h).requires_grad_()
    f = (torch.zeros_like(h) if zero else torch.randn_like(h)).requires_grad_()
    moments = torch.tensor([.2,.1],device='cuda',dtype=torch.float64,requires_grad=True)
    coeff = torch.tensor([.7,.3,.2],device='cuda',dtype=torch.float64,requires_grad=True)
    def fn(h,v,m,f,coeff):
        return approximate_momentum_step(h,v,m,f,*coeff.unbind(),.1)
    assert torch.autograd.gradcheck(fn,(h,v,moments,f,coeff),eps=1e-6,atol=2e-6,rtol=1e-4)
    assert torch.autograd.gradgradcheck(fn,(h,v,moments,f,coeff),eps=1e-6,atol=4e-6,rtol=1e-4)


def test_independent_basis_second_moment_oracle():
    # Deterministic orthogonal innovations: moment assumptions hold exactly.
    d = 128
    basis = torch.eye(d,device='cuda',dtype=torch.float64)*d**.5
    h, v = basis[0:1], .02*basis[1:2]
    moments = h.new_tensor([.02**2,0.])
    coeff = h.new_tensor([.8,.5,.4])
    naive_h, naive_v = h.clone(), v.clone()
    naive_q = moments[0]
    for i in range(64):
        force = basis[i+2:i+3]
        h, v, moments = approximate_momentum_step(h,v,moments,force,*coeff.unbind(),0.)
        observed = torch.stack((v.square().mean(),(h*v).mean()))
        torch.testing.assert_close(moments,observed,rtol=2e-12,atol=2e-12)
        torch.testing.assert_close(h.square().mean(),h.new_tensor(1.),rtol=2e-12,atol=2e-12)
        beta,gamma,eta = coeff
        naive_q = beta**2*naive_q+gamma**2
        naive_v = beta*naive_v+gamma*force
        naive_h = (naive_h+eta*naive_v)/torch.sqrt(1+eta**2*naive_q)
    # Confirms that dropping h-v covariance is a substantive mathematical error.
    assert naive_h.square().mean() > 2


def test_an_zero_momentum_matches_scalar_lerp_correction():
    h, f = [torch.randn(4,128,device='cuda',dtype=torch.float64) for _ in range(2)]
    eta = h.new_tensor(.3); gamma = h.new_tensor(1.)
    y, _, _ = approximate_momentum_step(h,torch.zeros_like(h),h.new_zeros(2),
                                       f,h.new_tensor(0.),gamma,eta,.1)
    z = f/(f.square().mean(-1,keepdim=True)+.01).sqrt()
    alpha = eta/(1+eta)
    expected = ((1-alpha)*h+alpha*z)/torch.sqrt((1-alpha)**2+alpha**2)
    torch.testing.assert_close(y,expected,rtol=1e-12,atol=1e-12)


def test_an_keeps_velocity_and_does_not_force_hidden_onto_sphere():
    h = torch.ones(2,128,device='cuda',dtype=torch.float64)
    v = .2*h; coeff = h.new_tensor([.7,.5,.3])
    y, u, _ = approximate_momentum_step(h,v,h.new_tensor([.04,.2]),h,*coeff.unbind(),.1)
    expected_u = .7*v+.5*h/(1.01**.5)
    torch.testing.assert_close(u,expected_u,rtol=1e-12,atol=1e-12)
    assert not torch.allclose(u, (y-h)/.3)
    assert not torch.allclose(y.square().mean(-1),h.new_ones(2))


def test_32_layer_finite_forward_backward():
    c = ConnectionConfig(depth=32,width=128,heads=1,sequence_length=32,vocab_size=256,
                         streams=1,residual_dynamics='an_heavy_ball')
    model = make(c); x,y = inputs()
    loss = model(x,y,dense=True); loss.backward()
    assert torch.isfinite(loss)
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    h, layers = model.backbone(model.wte(x),return_layers=True,token_ids=x)
    assert layers.shape == (2,16,32,128) and torch.isfinite(layers).all()
    assert layers.float().square().mean(-1).sqrt().max() < 10
