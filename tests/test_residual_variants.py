"""GPU checks for the authorized residual ablation configurations."""
from dataclasses import asdict, replace
import os

import pytest
import torch

from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for
from nanochat.engine import KVCache
from nanochat.packed_training import seed_all, snapshot, restore_rng
from nanochat.riemannian_momentum import sphere_project, tangent_project
from test_connection_training_resume import equal_tree
from test_riemannian_momentum import gpu, inputs

PRESETS = {'riemannian_embed': ('riemannian', 'token'),
           'heavy_ball_embed': ('heavy_ball', 'token'),
           'heavy_ball_zero': ('heavy_ball', 'zero'), 'sphere': ('sphere', 'zero')}
SELECTED = os.environ.get('RMT_EXPERIMENT')


@pytest.fixture(params=[SELECTED] if SELECTED in PRESETS else list(PRESETS))
def setting(request):
    mode, init = PRESETS[request.param]
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
    force = torch.randn_like(h)
    update = model.blocks[0].attn_residual
    actual, carried = update(h, v, force)
    if setting.residual_mode == 'sphere':
        assert v is None and carried is None and model.velocity_embedding is None
        alpha, = update.coefficients()
        expected = sphere_project(h + alpha*tangent_project(h, force))
    elif setting.residual_mode == 'heavy_ball':
        if setting.velocity_init == 'token':
            torch.testing.assert_close(v, model.velocity_embedding(x).float(), rtol=0, atol=0)
        else:
            assert model.velocity_embedding is None and torch.count_nonzero(v) == 0
        beta, gamma, eta = update.coefficients()
        u = beta*v + gamma*force
        expected = h + eta*u
        torch.testing.assert_close(carried, u, rtol=0, atol=0)
    else:
        assert v.abs().sum() > 0
        assert ((h*v).sum(-1).abs()/(h.norm(dim=-1)*v.norm(dim=-1))).max() < 1e-6
        beta, gamma, eta = update.coefficients()
        u = beta*v + gamma*tangent_project(h, force)
        expected = sphere_project(h + eta*u)
        assert ((actual*carried).sum(-1).abs()/(actual.norm(dim=-1)*carried.norm(dim=-1))).max() < 1e-6
        torch.testing.assert_close(carried.norm(dim=-1), u.norm(dim=-1), rtol=1e-6, atol=1e-6)
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
def test_zero_embedding_limit_and_sphere_identity(setting):
    model=make(setting, perturb=False).eval(); x,_=inputs()
    h,v=model.initial_state(model.wte(x),x)
    if setting.residual_mode=='sphere':
        top=model.backbone(model.wte(x),token_ids=x)
        torch.testing.assert_close(top,h,rtol=2e-6,atol=2e-6)
        assert all('velocity' not in n for n,_ in model.named_parameters())
    elif setting.velocity_init=='zero':
        top=model.backbone(model.wte(x),token_ids=x)
        torch.testing.assert_close(top,h,rtol=0,atol=0)
        assert torch.count_nonzero(v)==0 and model.velocity_embedding is None
        base=make(replace(setting,residual_dynamics='vanilla'),perturb=False).eval()
        base.load_state_dict({n:t for n,t in model.state_dict().items() if '_residual.' not in n})
        torch.testing.assert_close(model(x),base(x),rtol=0,atol=0)
    else:
        before=model(x)
        model.velocity_embedding.weight.zero_()
        zero=make(replace(setting,velocity_init='zero'),perturb=False).eval()
        zero.load_state_dict({n:t for n,t in model.state_dict().items() if n!='velocity_embedding.weight'})
        after=model(x)
        assert not torch.equal(before,after)
        torch.testing.assert_close(after,zero(x),rtol=0,atol=0)
