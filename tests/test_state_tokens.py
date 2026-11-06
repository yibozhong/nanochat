"""Independent attention oracle, state semantics, and end-to-end depth checks."""
from dataclasses import asdict
import copy
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for
from nanochat.connection_experiment import match_vanilla_initialization
from nanochat.state_tokens import StateTokenResidual
from nanochat.gpt import norm
from nanochat.engine import KVCache


@pytest.fixture(autouse=True)
def isolated_compile_cache():
    # Each case compiles different models/shapes/backends. Dynamo caches the
    # functools.partial wrapper globally; previous cases must not consume the
    # next case's specialization budget. Keep production limits unchanged.
    torch._dynamo.reset()
    try:
        yield
    finally:
        torch._dynamo.reset()


def record(name, data):
    root = Path('results/state-checks'); root.mkdir(parents=True, exist_ok=True)
    (root / (name + '.json')).write_text(json.dumps(data, indent=2) + '\n')


def oracle_attention(module, query, memory):
    # Explicit independent per-token/per-head calculation in FP64.
    batch, time, nq, width = query.shape
    hd = width // module.heads
    q = F.linear(query, module.q.weight).reshape(batch, time, nq, module.heads, hd)
    k = F.linear(memory, module.k.weight).reshape(batch, time, -1, module.heads, hd)
    v = F.linear(memory, module.v.weight).reshape(batch, time, -1, module.heads, hd)
    weights = torch.einsum('btihe,btjhe->bthij', q, k).div(hd**.5).softmax(-1)
    y = torch.einsum('bthij,btjhe->btihe', weights, v).reshape(batch, time, nq, width)
    return F.linear(y, module.out.weight)


@pytest.mark.parametrize('backend', ['math', 'sdpa'])
def test_all_gradients_against_independent_double_oracle(backend):
    torch.manual_seed(17)
    m = StateTokenResidual(32, 2, backend).double()
    for p in m.parameters():
        with torch.no_grad(): p.normal_(std=.2)
    h = torch.randn(2, 3, 32, dtype=torch.float64, requires_grad=True)
    s = torch.randn(2, 3, 4, 32, dtype=torch.float64, requires_grad=True)
    a, b = m(h, s)
    ar = h + oracle_attention(m.read, h.unsqueeze(-2), s).squeeze(-2)
    br = s + oracle_attention(m.write, s, torch.cat((s, h.unsqueeze(-2)), -2))
    weights = (torch.randn_like(a), torch.randn_like(b))
    params = (h, s, *m.read.parameters(), *m.write.parameters())
    ga = torch.autograd.grad((a, b), params, weights, retain_graph=True)
    gb = torch.autograd.grad((ar, br), params, weights)
    for x, y in zip((a, b, *ga), (ar, br, *gb)):
        torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-11)


def test_add_old_state_and_token_locality():
    torch.manual_seed(12)
    m = StateTokenResidual(32, 2, 'math').double()
    h = torch.randn(2, 7, 32, dtype=torch.float64, requires_grad=True)
    s = torch.randn(2, 7, 4, 32, dtype=torch.float64, requires_grad=True)
    before = s.detach().clone()
    a, b = m(h, s)
    altered = s.detach().clone(); altered[0, 3] += 10
    aa, bb = m(h, altered)
    mask = torch.ones(2, 7, dtype=torch.bool); mask[0, 3] = False
    assert torch.equal(a[mask], aa[mask]) and torch.equal(b[mask], bb[mask])
    gs, = torch.autograd.grad(a[0, 2].sum()+b[0, 2].sum(), s)
    mask[:] = True; mask[0, 2] = False
    assert torch.count_nonzero(gs[mask]) == 0
    assert torch.equal(s, before), 'Write mutated the pre-update state'
    with torch.no_grad(): m.write.v.weight.zero_()
    after_read, replaced = m(h, s)
    assert torch.equal(after_read, a)
    assert torch.equal(replaced, s), 'Zero write must preserve the old state exactly'


def test_initial_information_and_gradient_retention():
    torch.manual_seed(42)
    m = StateTokenResidual(512, 4, 'math')
    e = norm(torch.randn(2, 32, 512)).requires_grad_()
    s = m.initialize(e)
    assert torch.count_nonzero(m.slots.mean(0)) == 0
    torch.testing.assert_close(s.mean(-2), e, rtol=1e-6, atol=3e-7)
    rows = []
    probe = torch.randn_like(e)
    for step in range(16):
        h, s = m(torch.zeros_like(e), s)
        relative = ((h-e).norm()/e.norm()).item()
        diversity = (s-s.mean(-2, keepdim=True)).square().mean().sqrt().item()
        rows.append(dict(sublayer=step+1, relative_embedding_error=relative, slot_rms=diversity))
    grad, = torch.autograd.grad(h, e, probe)
    grad_error = ((grad-probe).norm()/probe.norm()).item()
    record('initialization', dict(layers=rows, gradient_relative_error=grad_error))
    assert rows[-1]['relative_embedding_error'] < .1, rows
    assert rows[-1]['slot_rms'] > .25, rows
    assert grad_error < .2, grad_error


def test_shared_parameters_backbone_optimizer_and_metadata():
    c = ConnectionConfig(variant='state_tokens', width=32, heads=2, depth=3,
                         vocab_size=64, state_attention_backend='math')
    torch.manual_seed(42); m = build_connection_model(c)
    match_vanilla_initialization(m, 42)
    names = dict(m.named_parameters())
    assert len([n for n in names if n.startswith('state_residual.')]) == 9
    assert not any('_connection.' in n for n in names)
    assert sum(p.numel() for p in m.state_residual.parameters()) == 8*32*32+4*32
    assert len({id(b.attn.c_q.weight) for b in m.blocks}) == 3
    assert m.state_residual.read.q.weight is not m.state_residual.write.q.weight
    opt, _ = optimizer_for(m)
    ids = [id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids) == len(set(ids)) == len(names)
    assert ConnectionConfig.from_checkpoint(asdict(c)) == c
    obsolete = asdict(c); obsolete.pop('state_write_residual')
    with pytest.raises(ValueError, match='retired'): ConnectionConfig.from_checkpoint(obsolete)
    assert torch.count_nonzero(m.state_residual.write.out.weight) == 0
    old = asdict(ConnectionConfig()); old.pop('state_attention_backend')
    assert ConnectionConfig.from_checkpoint(old) == ConnectionConfig()
    for kwargs in (dict(architecture='native'), dict(streams=3), dict(fused_connection_grad=True)):
        with pytest.raises(ValueError): ConnectionConfig(variant='state_tokens', **kwargs)


def randomize_branches(m):
    with torch.no_grad():
        m.state_residual.write.out.weight.normal_(std=.01)
        for b in m.blocks:
            b.attn.c_proj.weight.normal_(std=.02)
            b.mlp.c_proj.weight.normal_(std=.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU implementation gate')
@pytest.mark.parametrize('backend', ['auto', 'sdpa', 'triton'])
def test_slot_attention_bf16_vjp(backend):
    torch.manual_seed(17)
    m = StateTokenResidual(128, 2, backend).cuda()
    with torch.no_grad():
        for p in m.parameters(): p.normal_(std=.08)
    ref = copy.deepcopy(m).double()
    ref.read.backend = ref.write.backend = 'math'
    with torch.no_grad():
        for p, r in zip(m.parameters(), ref.parameters()): r.copy_(p.bfloat16().double())
    h = torch.randn(2, 37, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    s = torch.randn(2, 37, 4, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    hr = h.detach().double().requires_grad_(); sr = s.detach().double().requires_grad_()
    a = m(h, s); b = ref(hr, sr)
    probe = tuple(torch.randn_like(x) for x in a)
    ga = torch.autograd.grad(a, (h, s, *m.read.parameters(), *m.write.parameters()), probe)
    gb = torch.autograd.grad(b, (hr, sr, *ref.read.parameters(), *ref.write.parameters()),
                             tuple(x.double() for x in probe))
    errors = [((x.double()-y).norm()/y.norm().clamp_min(1e-12)).item()
              for x, y in zip((*a, *ga), (*b, *gb))]
    record('slot-vjp-'+backend, dict(relative_errors=errors))
    assert max(errors) < .025, errors


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU implementation gate')
@pytest.mark.parametrize('backend', ['auto', 'triton'])
def test_bf16_initial_state_depth_retention(backend):
    torch.manual_seed(42)
    m = StateTokenResidual(512, 4, backend).cuda()
    e = norm(torch.randn(2, 32, 512, device='cuda', dtype=torch.bfloat16)).requires_grad_()
    state = m.initialize(e)
    probe = torch.randn_like(e)
    for _ in range(16): h, state = m(torch.zeros_like(e), state)
    gradient, = torch.autograd.grad(h, e, probe)
    result = dict(embedding_error=((h-e).float().norm()/e.float().norm()).item(),
        slot_rms=(state.float()-state.float().mean(-2,keepdim=True)).square().mean().sqrt().item(),
        gradient_error=((gradient-probe).float().norm()/probe.float().norm()).item())
    record('bf16-initialization-'+backend, result)
    assert result['embedding_error']<.1 and result['slot_rms']>.25 and result['gradient_error']<.2, result


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU implementation gate')
@pytest.mark.parametrize('backend', ['auto', 'triton'])
def test_2k_causal_cache_cce_compile_recompute(backend):
    torch.manual_seed(42)
    c = ConnectionConfig(variant='state_tokens', width=128, heads=2, depth=2, vocab_size=128,
                         state_attention_backend=backend)
    m = build_connection_model(c).prepare()
    randomize_branches(m)
    ids = torch.randint(0, 128, (2, 2048), device='cuda')
    changed = ids.clone(); changed[0, 1024:] = (changed[0, 1024:]+19) % 128
    with torch.no_grad():
        a, b = m(ids), m(changed)
        assert torch.equal(a[0, :1024], b[0, :1024]) and torch.equal(a[1], b[1])
        again = m(ids)
        assert torch.equal(a, again), 'Runtime states persisted between independent forwards'
        cache = KVCache(2, 2, 32, 64, 2, 'cuda', torch.bfloat16)
        pre = m(ids[:, :7], kv_cache=cache)
        nxt = m(ids[:, 7:8], kv_cache=cache)
        full = m(ids[:, :8])
        torch.testing.assert_close(pre, full[:, :7], atol=.004, rtol=.03)
        torch.testing.assert_close(nxt, full[:, 7:], atol=.004, rtol=.03)
    x, y = ids[:, :64].contiguous(), ids[:, 1:65].contiguous()
    loss = m(x, y); dense = m(x, y, dense=True)
    torch.testing.assert_close(loss, dense, atol=2e-4, rtol=2e-4)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters()}
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert all(grads[n].norm()>0 for n in grads if n.startswith('state_residual.'))
    m.zero_grad(set_to_none=True); m.prepare(compile_blocks=True)
    compiled = m(x, y); compiled.backward()
    torch.testing.assert_close(loss, compiled, rtol=2e-5, atol=2e-5)
    errors = {n: ((p.grad-grads[n]).float().norm()/grads[n].float().norm().clamp_min(1e-9)).item()
              for n, p in m.named_parameters()}
    assert max(errors.values()) < .03, errors
    m.zero_grad(set_to_none=True); m.config.activation_checkpoint = True
    recomputed = m(x, y); recomputed.backward()
    torch.testing.assert_close(compiled, recomputed, rtol=2e-5, atol=2e-5)
    errors2 = {n: ((p.grad-grads[n]).float().norm()/grads[n].float().norm().clamp_min(1e-9)).item()
               for n, p in m.named_parameters()}
    assert max(errors2.values()) < .03, errors2
    record('model-'+backend, dict(ce=loss.item(), compile_gradient_errors=errors,
                        recompute_gradient_errors=errors2, causal_tokens=2048))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU implementation gate')
def test_full_model_triton_matches_compiled_torch():
    torch.manual_seed(42)
    c = ConnectionConfig(variant='state_tokens', width=128, heads=2, depth=2,
                         vocab_size=128, state_attention_backend='triton')
    model = build_connection_model(c).prepare()
    randomize_branches(model)
    other = copy.deepcopy(model)
    other.config.state_attention_backend = 'auto'
    other.state_residual.read.backend = other.state_residual.write.backend = 'auto'
    model.prepare(compile_blocks=True); other.prepare(compile_blocks=True)
    x = torch.randint(0, 128, (2, 128), device='cuda')
    a = model(x[:, :-1], x[:, 1:]); b = other(x[:, :-1], x[:, 1:])
    a.backward(); b.backward()
    torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
    errors = {name: ((p.grad-q.grad).float().norm()/q.grad.float().norm().clamp_min(1e-9)).item()
              for (name,p),q in zip(model.named_parameters(),other.parameters())}
    assert max(errors.values()) < .03, errors
    record('full-model-backends', dict(triton_ce=a.item(), torch_ce=b.item(), gradient_errors=errors))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU implementation gate')
def test_model_optimizer_checkpoint_fixed_next_update(tmp_path):
    torch.manual_seed(42)
    c = ConnectionConfig(variant='state_tokens', width=128, heads=2, depth=2, vocab_size=128,
                         state_attention_backend='triton')
    m = build_connection_model(c).prepare(); opt, _ = optimizer_for(m)
    ids = torch.randint(0, 128, (2, 33), device='cuda')
    m(ids[:, :-1], ids[:, 1:]).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    path = tmp_path/'state.pt'
    torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(), config=asdict(c)), path)
    saved = torch.load(path, weights_only=False)
    n = build_connection_model(ConnectionConfig.from_checkpoint(saved['config'])).prepare()
    opt2, _ = optimizer_for(n)
    n.load_state_dict(saved['model']); opt2.load_state_dict(saved['optimizer'])
    for a, b in zip(m.parameters(), n.parameters()): assert torch.equal(a, b)
    for a, b in zip(m.parameters(), n.parameters()):
        grad = torch.randn_like(a)
        a.grad = grad.clone(); b.grad = grad.clone()
    opt.step(); opt2.step()
    for a, b in zip(m.parameters(), n.parameters()): assert torch.equal(a, b)
