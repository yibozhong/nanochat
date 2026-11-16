"""GPU-only Form correctness, state migration and resume checks."""
import copy
from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

from nanochat import optim as optim_module
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.connection_experiment import configure_attention
from nanochat.connection_cost import model_cost
from nanochat.form import FormController, VirtualGate, unpack_optimizer
from nanochat.packed_training import seed_all, snapshot, restore_rng
from test_connection_training_resume import equal_tree


@pytest.fixture(autouse=True)
def setup():
    assert torch.cuda.is_available(), 'Form numerical tests must run on a GPU worker'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    configure_attention('fa2')
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.fixture
def eager_optimizer(monkeypatch):
    # Exercise the exact optimizer math without compiling dozens of tiny
    # diagnostic shapes. Production kernels are tested separately below and
    # by the real-shape preflight and CLI restart check.
    for name in ('muon_step_fused', 'adamw_step_fused'):
        fn = getattr(optim_module, name)
        monkeypatch.setattr(optim_module, name, fn._torchdynamo_orig_callable)


def build(mode='taylor', compiled=False, total_steps=44, config=None):
    seed_all(42)
    c = config or ConnectionConfig(depth=3, width=64, heads=1, sequence_length=32,
        vocab_size=128, streams=1, form_mode=mode)
    model = ConnectionModel(c).prepare(compile_blocks=compiled)
    opt, wd = optimizer_for(model)
    form = FormController(model, total_steps) if mode != 'none' else None
    return model, opt, wd, form


def batch():
    x = torch.arange(64, device='cuda').reshape(2, 32) % 128
    return x, (x + 1) % 128


def update(model, opt, form, step, total=44):
    x, y = batch()
    schedule(opt, step, total, .01)
    opt.zero_grad(set_to_none=True)
    if form is not None:form.begin_step()
    loss = model(x, y, dense=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and bool(p.grad.isfinite().all()) for p in model.parameters())
    if form is not None:form.observe()
    opt.step()
    return loss.item()


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float64])
def test_virtual_gate_gradient(dtype):
    x = torch.randn(2, 5, 17, device='cuda', dtype=dtype, requires_grad=True)
    g = torch.ones(17, device='cuda', dtype=torch.float64 if dtype == torch.float64 else torch.float32, requires_grad=True)
    dy = torch.randn_like(x)
    out = VirtualGate.apply(x, g)
    dx, dg = torch.autograd.grad(out, (x, g), dy)
    assert torch.equal(out, x) and torch.equal(dx, dy)
    expected = (x.to(g.dtype) * dy.to(g.dtype)).sum((0, 1))
    torch.testing.assert_close(dg, expected, rtol=0, atol=0)


def test_gate_double_gradcheck():
    x = torch.randn(2, 3, device='cuda', dtype=torch.float64, requires_grad=True)
    g = torch.ones(3, device='cuda', dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(VirtualGate.apply, (x, g))


def test_disabled_and_enabled_initial_function_and_optimizer_match(eager_optimizer):
    base, bo, _, _ = build('none')
    model, opt, _, form = build()
    for (n, p), (other, q) in zip(base.named_parameters(), model.named_parameters()):
        assert n == other and torch.equal(p, q)
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in model.parameters())
    assert not any('gate' in n for n, _ in model.named_parameters())
    for step in range(3):
        x, y = batch()
        with torch.no_grad():
            torch.testing.assert_close(base(x, y, dense=True), model(x, y, dense=True), rtol=0, atol=0)
        b = update(model, opt, form, step)
        # Same gradient input verifies that enabling statistics does not change
        # the native optimizer update, independently of FA2 backward atomics.
        for p, q in zip(base.parameters(), model.parameters()):p.grad = q.grad.clone()
        schedule(bo, step, 44, .01);bo.step()
        for p, q in zip(base.parameters(), model.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert all(torch.equal(m.virtual_gate, torch.ones_like(m.virtual_gate)) for m in form.mlps)


def test_accumulated_gate_gradient_ema_once_per_update(eager_optimizer):
    model, opt, _, form = build()
    update(model, opt, form, 0)
    beta = model.config.form_ema_decay
    previous = [(m.form_mean.clone(), m.form_second.clone()) for m in form.mlps]
    x, y = batch()
    opt.zero_grad(set_to_none=True);form.begin_step()
    for i in range(2):
        (model(x[i:i+1], y[i:i+1], dense=True) / 2).backward()
    gradients = [m.virtual_gate.grad.clone() for m in form.mlps]
    form.observe()
    assert form.window_steps == 2
    for m, g, (mean, second) in zip(form.mlps, gradients, previous):
        torch.testing.assert_close(m.form_mean, mean.lerp(g, 1-beta))
        torch.testing.assert_close(m.form_second, second.lerp(g.square(), 1-beta))


def test_nonfinite_gate_rejected(eager_optimizer):
    model, _, _, form = build()
    for m in form.mlps:m.virtual_gate.grad = torch.zeros_like(m.virtual_gate)
    form.mlps[0].virtual_gate.grad[0] = float('nan')
    with pytest.raises(FloatingPointError):form.observe()
    assert form.window_steps == 0


def check_surgery(model, opt, form, step, loss):
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    old_slots, old_settings = unpack_optimizer(model, opt)
    old_slots = copy.deepcopy(old_slots)
    old_cost = model_cost(model.config)
    opt, report = form.apply(opt, step, loss)
    plan = report['plan']
    slots, settings = unpack_optimizer(model, opt)
    assert settings == old_settings
    for l, (keep, growth) in enumerate(zip(plan['keep'], plan['growth'])):
        mlp = model.blocks[l].mlp
        for name, axis in (('c_fc', 0), ('c_proj', 1)):
            key = f'blocks.{l}.mlp.{name}.weight'
            p = dict(model.named_parameters())[key]
            idx = torch.tensor(keep, device=p.device)
            torch.testing.assert_close(p.narrow(axis, 0, len(keep)), before[key].index_select(axis, idx), rtol=0, atol=0)
            for slot, value in slots[key].items():
                torch.testing.assert_close(value.narrow(axis, 0, len(keep)), old_slots[key][slot].index_select(axis, idx), rtol=0, atol=0)
                if growth:assert torch.count_nonzero(value.narrow(axis, len(keep), growth)) == 0
        if growth:
            assert torch.count_nonzero(mlp.c_proj.weight[:, len(keep):]) == 0
            added = mlp.c_fc.weight[len(keep):]
            assert added.abs().max() <= .4 * (3 / model.config.width) ** .5
            assert added.std() > 0
        # Growth preserves the pruned FFN function, with no compensation.
        x = torch.randn(2, 7, model.config.width, device='cuda')
        up = before[f'blocks.{l}.mlp.c_fc.weight'][keep]
        down = before[f'blocks.{l}.mlp.c_proj.weight'][:, keep]
        expected = F.linear(F.relu(F.linear(x, up)).square(), down)
        torch.testing.assert_close(mlp(x), expected, rtol=2e-5, atol=2e-6)
        assert torch.count_nonzero(mlp.form_mean) == torch.count_nonzero(mlp.form_second) == 0
    for name, p in model.named_parameters():
        if '.mlp.' not in name:
            assert torch.equal(before[name], p)
            equal_tree(old_slots[name], slots[name])
    assert model_cost(model.config) == old_cost
    assert form.window_steps == 0
    return opt, report


def test_prune_grow_preserves_weights_slots_and_function(eager_optimizer):
    model, opt, _, form = build()
    for step in range(4):loss = update(model, opt, form, step)
    opt, report = check_surgery(model, opt, form, 4, loss)
    assert sum(report['event']['pruned']) == sum(report['event']['grown']) == 64
    update(model, opt, form, 4)


def test_minimum_width_preserves_down_projection_variance_axis(eager_optimizer):
    model, opt, _, form = build()
    for step in range(12):
        loss = update(model, opt, form, step)
        if form.due(step + 1):
            # Channel scores deliberately drain layer zero to exactly hidden_dim.
            for l, m in enumerate(form.mlps):
                m.form_mean.fill_(10 - l * 10);m.form_second.zero_()
            opt, _ = check_surgery(model, opt, form, step + 1, loss)
    assert model.config.ffn_widths[0] == model.config.width
    slots, _ = unpack_optimizer(model, opt)
    assert slots['blocks.0.mlp.c_proj.weight']['second_momentum_buffer'].shape == (1, 64)
    update(model, opt, form, 12)


@pytest.mark.parametrize('mode', ['taylor', 'random'])
def test_all_ten_windows_and_causality(mode, eager_optimizer):
    model, opt, _, form = build(mode)
    for step in range(44):
        loss = update(model, opt, form, step)
        form.record_next_loss(step + 1, loss)
        if form.due(step + 1):
            opt, _ = form.apply(opt, step + 1, loss)
            assert sum(model.config.ffn_widths) == 768
            assert min(model.config.ffn_widths) >= 64
    assert form.completed == 10 and len(form.history) == 10
    assert all('next_train_ce' in h for h in form.history)
    x, _ = batch();changed = x.clone();changed[:, 16:] = (changed[:, 16:] + 15) % 128
    model.eval()
    with torch.no_grad():
        a, b = model(x), model(changed)
    torch.testing.assert_close(a[:, :16], b[:, :16], rtol=0, atol=0)


@pytest.mark.parametrize('boundary', [3, 4, 5, 40])
def test_checkpoint_dynamic_shapes_stats_rng_and_next_update(boundary, tmp_path, eager_optimizer):
    model, opt, wd, form = build()
    for step in range(boundary):
        loss = update(model, opt, form, step)
        if form.due(step + 1):opt, _ = form.apply(opt, step + 1, loss)
    path = tmp_path / 'checkpoint.pt'
    torch.save(snapshot(model, opt, boundary, config=asdict(model.config), form=form.state_dict()), path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    other, other_opt, other_wd, other_form = build(config=ConnectionConfig.from_checkpoint(state['config']))
    other.load_state_dict(state['model']);other_opt.load_state_dict(state['optimizer'])
    other_form.load_state_dict(state['form'], boundary)
    assert wd == other_wd
    equal_tree(model.state_dict(), other.state_dict())
    equal_tree(opt.state_dict(), other_opt.state_dict())
    equal_tree(form.state_dict(), other_form.state_dict())
    restore_rng(state['rng']);expected = torch.rand(9, device='cuda')
    restore_rng(state['rng']);assert torch.equal(expected, torch.rand(9, device='cuda'))
    # Same next gradients, scheduler and state must give bit-identical updates.
    for target, optimizer in ((model, opt), (other, other_opt)):
        for i, p in enumerate(target.parameters()):p.grad = torch.full_like(p, (i+1)*1e-4)
        schedule(optimizer, boundary, 44, wd);optimizer.step()
    equal_tree(model.state_dict(), other.state_dict())
    equal_tree(opt.state_dict(), other_opt.state_dict())
    # Isolated new-channel RNG and global decisions must also survive resume.
    while not form.due(boundary):
        update(model, opt, form, boundary)
        other_opt.zero_grad(set_to_none=True);other_form.begin_step()
        for p, q in zip(other.parameters(), model.parameters()):p.grad = q.grad.clone()
        for p, q in zip(other_form.mlps, form.mlps):p.virtual_gate.grad = q.virtual_gate.grad.clone()
        other_form.observe();schedule(other_opt, boundary, 44, .01);other_opt.step()
        boundary += 1
        if boundary >= 44:return
    opt, a = form.apply(opt, boundary, 1.)
    other_opt, b = other_form.apply(other_opt, boundary, 1.)
    assert a == b
    equal_tree(model.state_dict(), other.state_dict())
    equal_tree(opt.state_dict(), other_opt.state_dict())


@pytest.mark.parametrize('checkpoint', [False, True])
def test_compiled_real_optimizer_gate_and_resize(checkpoint):
    c = ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32,
        vocab_size=128, streams=1, form_mode='taylor', activation_checkpoint=checkpoint)
    model, opt, _, form = build(compiled=True, total_steps=22, config=c)
    for step in range(3):
        loss = update(model, opt, form, step, total=22)
        if form.due(step + 1):opt, _ = form.apply(opt, step + 1, loss)
    assert form.completed == 1 and form.window_steps == 1
    assert all(m.virtual_gate.grad is not None and m.virtual_gate.is_leaf for m in form.mlps)
    assert any(torch.count_nonzero(m.form_mean) for m in form.mlps)


@pytest.mark.parametrize('checkpoint', [False, True])
def test_compiled_gate_and_weight_gradients_match_reference(checkpoint):
    c = ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32,
        vocab_size=128, streams=1, form_mode='taylor', activation_checkpoint=checkpoint)
    model, _, _, form = build(compiled=True, total_steps=22, config=c)
    with torch.no_grad():
        for block in model.blocks:
            block.attn.c_proj.weight.normal_(std=.02)
            block.mlp.c_proj.weight.normal_(std=.02)
    ref, _, _, ref_form = build(total_steps=22, config=ConnectionConfig(**asdict(c)))
    ref.load_state_dict(model.state_dict())
    x, y = batch()
    a = model(x, y, dense=True);a.backward()
    b = ref(x, y, dense=True);b.backward()
    torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
    pairs = [(p.grad, q.grad) for p, q in zip(model.parameters(), ref.parameters())]
    pairs += [(p.virtual_gate.grad, q.virtual_gate.grad) for p, q in zip(form.mlps, ref_form.mlps)]
    for actual, expected in pairs:
        error = (actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-10)
        assert error < .04, error.item()


def test_legacy_config_and_validation():
    legacy = ConnectionConfig.from_checkpoint(dict(variant='baseline', depth=3, width=64, heads=1))
    assert legacy.form_mode == 'none' and legacy.ffn_widths is None
    for kwargs in (dict(form_mode='taylor', latent_feedback=True),
                   dict(form_mode='taylor', ffn_widths=[2048]*7+[2047]),
                   dict(form_mode='taylor', form_ema_decay=1.),
                   dict(form_mode='taylor', form_lambda=float('nan'))):
        with pytest.raises(ValueError):ConnectionConfig(**kwargs)
