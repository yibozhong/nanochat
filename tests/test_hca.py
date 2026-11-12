"""Run on a Condor worker, never on the AP: pytest -q tests/test_hca.py."""
import copy
from dataclasses import asdict, replace
import io

import pytest
import torch
from torch.nn import functional as F

from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for
from nanochat.gpt import apply_rotary_emb, norm
from nanochat.hca import HCAKVCache, boundary_step, pool_step


DEVICE = 'cuda'


def config(mode='dynamic_hca', **kwargs):
    opts = dict(variant='baseline', streams=1, depth=2, width=32, heads=2,
                sequence_length=80, vocab_size=64, attention_mode=mode,
                hca_block_size=4, hca_budget_chunks=4, hca_window=16, hca_noise=False)
    opts.update(kwargs)
    return ConnectionConfig(**opts)


def model(c=None, dtype=torch.float32):
    torch.manual_seed(81)
    m = build_connection_model(c or config()).prepare(DEVICE, dtype=dtype)
    with torch.no_grad():
        for b in m.blocks:
            # Exercise attention/MLP paths, not the initial zero branch output.
            b.attn.c_proj.weight.normal_(std=.08)
            b.mlp.c_proj.weight.normal_(std=.03)
            if hasattr(b.attn, 'pool_score'):
                b.attn.pool_score.weight.normal_(std=.1)
                if b.attn.boundary is not None:
                    b.attn.boundary.weight.normal_(std=.5)
    return m


def independent_ends(content, c):
    result = torch.zeros_like(content)
    for row in range(content.shape[0]):
        age = used = 0
        for t, value in enumerate(content[row].detach().cpu().tolist()):
            offset = t % (c.hca_block_size * c.hca_budget_chunks)
            if c.attention_mode == 'fixed_hca':
                end = (t + 1) % c.hca_block_size == 0
            else:
                remaining = c.hca_budget_chunks - used
                tokens_left = c.hca_block_size * c.hca_budget_chunks - offset
                end = remaining == tokens_left or tokens_left == 1
                if not end and remaining > 1:
                    end = value + (age + 1 - c.hca_block_size) / max(c.hca_block_size / 4, 1) >= 0
            result[row, t] = float(end)
            used += int(end)
            age = 0 if end else age + 1
            if offset == c.hca_block_size * c.hca_budget_chunks - 1:
                used = 0
    return result


def oracle(attn, args, cos_sin):
    """Independent hard partition + direct softmax pooling + explicit attention."""
    q, raw_k, local_k, v, score, content = args
    ends = independent_ends(content, attn.config)
    rows = []
    for b in range(q.shape[0]):
        chunks = []
        start = 0
        for end in ends[b].nonzero().flatten().tolist():
            weights = (10 * torch.tanh(score[b, start:end+1] / 10)).softmax(0)
            pk = (weights * raw_k[b, start:end+1]).sum(0)
            pv = (weights * v[b, start:end+1]).sum(0)
            cs = tuple(z[:, end:end+1] for z in cos_sin)
            pk = (norm(apply_rotary_emb(pk[None, None], *cs)) * 1.2)[0, 0]
            chunks.append((end, pk, pv))
            start = end + 1
        outputs = []
        for t in range(q.shape[1]):
            keys = list(local_k[b, max(0, t-attn.config.hca_window+1):t+1].unbind(0))
            vals = list(v[b, max(0, t-attn.config.hca_window+1):t+1].unbind(0))
            for end, pk, pv in chunks:
                if end < t:
                    keys.append(pk); vals.append(pv)
            keys, vals = torch.stack(keys), torch.stack(vals)
            scores = torch.einsum('hd,shd->hs', q[b, t], keys) / q.shape[-1] ** .5
            outputs.append(torch.einsum('hs,shd->hd', scores.softmax(-1), vals))
        rows.append(torch.stack(outputs))
    return torch.stack(rows), ends


@pytest.mark.parametrize('mode', ['fixed_hca', 'dynamic_hca'])
def test_fp64_oracle_all_continuous_vjps(mode):
    m = model(config(mode)).double().eval()
    a = m.blocks[0].attn
    x = torch.randn(2, 37, 32, device=DEVICE, dtype=torch.float64)
    cs = (m.cos[:, :37], m.sin[:, :37])
    args = tuple(z.detach().requires_grad_() for z in a.project(x, cs))
    actual, ends = a.dense(*args, cs)
    expected, ref_ends = oracle(a, args, cs)
    torch.testing.assert_close(ends, ref_ends, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    upstream = torch.randn_like(actual)
    # Boundary STE is deliberately biased; continuous inputs have exact VJPs.
    ga = torch.autograd.grad((actual * upstream).sum(), args[:5], retain_graph=True)
    gr = torch.autograd.grad((expected * upstream).sum(), args[:5])
    for a_grad, r_grad in zip(ga, gr):
        torch.testing.assert_close(a_grad, r_grad, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('mode', ['fixed_hca', 'dynamic_hca'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_prefill_split_decode_and_tail(mode, dtype):
    m = model(config(mode), dtype).eval()
    ids = torch.randint(0, 64, (2, 53), device=DEVICE)
    with torch.no_grad():
        reference = m(ids)
        cache = m.make_kv_cache(2)
        pieces, start = [], 0
        for count in [3, 1, 11, 2, 16, 7, 13]:
            pieces.append(m(ids[:, start:start+count], kv_cache=cache)); start += count
        actual = torch.cat(pieces, 1)
        single_cache = m.make_kv_cache(2)
        single = torch.cat([m(ids[:, i:i+1], kv_cache=single_cache) for i in range(53)], 1)
    tol = .004 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual, reference, rtol=tol, atol=tol)
    torch.testing.assert_close(single, reference, rtol=tol, atol=tol)
    assert cache.statistics()['completed_chunks'] == single_cache.statistics()['completed_chunks']
    for layer, other in zip(cache.layers, single_cache.layers):
        for state, ref in zip(layer, other):
            assert state.ends == ref.ends and state.age == ref.age
            assert state.age == 52 - (state.ends[-1] if state.ends else -1)


@pytest.mark.parametrize('mode', ['fixed_hca', 'dynamic_hca'])
def test_prefix_causality_and_future_gradients(mode):
    m = model(config(mode)).eval()
    ids = torch.randint(0, 64, (1, 40), device=DEVICE)
    altered = ids.clone(); altered[:, 19:] = (altered[:, 19:] + 1) % 64
    with torch.no_grad():
        torch.testing.assert_close(m(ids)[:, :19], m(altered)[:, :19], rtol=0, atol=0)
    a = m.blocks[0].attn
    x = torch.randn(1, 40, 32, device=DEVICE, requires_grad=True)
    y = a(x, None, (m.cos[:, :40], m.sin[:, :40]), (-1, 0), None)
    g, = torch.autograd.grad(y[:, :19].square().sum(), x)
    assert torch.count_nonzero(g[:, 19:]) == 0
    assert torch.count_nonzero(g[:, :19]) > 0


def test_exact_budget_even_adversarial_router():
    c = config(hca_block_size=16, hca_budget_chunks=16, hca_window=256, sequence_length=2048)
    for constant in (-1000., 0., 1000.):
        content = torch.full((3, 2048), constant, device=DEVICE)
        content[2] = torch.randn(2048, device=DEVICE) * 30
        age = torch.zeros(3, dtype=torch.long, device=DEVICE); used = age.clone()
        ends = []
        for t in range(2048):
            gate = boundary_step(content[:, t], age, used, t, c)
            hard = gate.bool(); ends.append(gate)
            age = torch.where(hard, 0, age+1); used += hard.long()
            if (t+1) % 256 == 0:
                assert (used == 16).all(); used.zero_()
        ends = torch.stack(ends, 1)
        torch.testing.assert_close(ends, independent_ends(content, c))
        assert (ends.sum(1) == 128).all()


def test_language_model_loss_trains_router_and_pool():
    m = model().train()
    x = torch.randint(0, 64, (2, 64), device=DEVICE)
    loss = m(x, (x+1) % 64, dense=True)
    loss.backward()
    for b in m.blocks:
        for p in (b.attn.boundary.weight, b.attn.pool_score.weight):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 1e-8
    assert torch.isfinite(loss)


def test_pool_extreme_scores_and_long_open_chunk():
    dtype = torch.float64
    k = torch.randn(1, 256, 2, 8, device=DEVICE, dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    score = (torch.randn_like(k) * 1000).requires_grad_()
    p = tuple(torch.zeros_like(k[:, 0]) for _ in range(3))
    for t in range(256):
        p, pk, pv = pool_step(k[:, t], v[:, t], score[:, t], p,
                             torch.tensor([float(t == 0)], device=DEVICE, dtype=dtype))
    w = (10 * torch.tanh(score/10)).softmax(1)
    torch.testing.assert_close(pk, (w*k).sum(1), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(pv, (w*v).sum(1), atol=1e-12, rtol=1e-12)
    for g in torch.autograd.grad(pk.sum()+pv.sum(), (k, v, score)):
        assert torch.isfinite(g).all()


def test_backbone_rng_and_legacy_config():
    configs = [config('full'), config('fixed_hca'), config('dynamic_hca')]
    models, states = [], []
    for c in configs:
        torch.manual_seed(123)
        models.append(build_connection_model(c))
        states.append(torch.get_rng_state())
    for m, state in zip(models[1:], states[1:]):
        torch.testing.assert_close(state, states[0], rtol=0, atol=0)
        for name, p in models[0].state_dict().items():
            torch.testing.assert_close(p, m.state_dict()[name], rtol=0, atol=0)
    legacy = asdict(config('full'))
    for key in list(legacy):
        if key.startswith('hca_') or key == 'attention_mode':
            del legacy[key]
    assert ConnectionConfig.from_checkpoint(legacy).attention_mode == 'full'


def test_cache_roundtrip_broadcast_reset_and_bad_api():
    m = model().eval()
    x = torch.randint(0, 64, (1, 38), device=DEVICE)
    cache = m.make_kv_cache()
    with torch.no_grad():
        m(x[:, :13], kv_cache=cache)
        buffer = io.BytesIO(); torch.save(cache.state_dict(), buffer); buffer.seek(0)
        restored = m.make_kv_cache(); restored.load_state_dict(torch.load(buffer, weights_only=False))
        expanded = m.make_kv_cache(3); expanded.prefill(restored)
        expected = m(x[:, 13:], kv_cache=cache)
        torch.testing.assert_close(m(x[:, 13:], kv_cache=restored), expected, rtol=0, atol=0)
        torch.testing.assert_close(m(x[:, 13:].expand(3, -1), kv_cache=expanded), expected.expand(3, -1, -1), atol=2e-6, rtol=2e-6)
    cache.reset()
    assert cache.statistics()['storage_bytes'] == 0 and cache.get_pos() == 0
    with pytest.raises(ValueError, match='no_grad'):
        m(x, kv_cache=cache)
    with pytest.raises(ValueError):
        m.prepare(DEVICE, compile_blocks=True)
    with pytest.raises(ValueError):
        replace(config(), hca_window=1)


def test_train_eval_hard_forward_checkpoint_and_rng_restore():
    c = config(hca_noise=True)
    m = model(c).train()
    other = copy.deepcopy(m); other.config = replace(c, activation_checkpoint=True)
    x = torch.randint(0, 64, (1, 48), device=DEVICE)
    rng = torch.cuda.get_rng_state()
    loss = m(x, (x+1) % 64, dense=True); loss.backward()
    torch.cuda.set_rng_state(rng)
    loss2 = other(x, (x+1) % 64, dense=True); loss2.backward()
    torch.testing.assert_close(loss, loss2, atol=0, rtol=0)
    for p, q in zip(m.parameters(), other.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    m.config.hca_noise = False
    with torch.no_grad():
        train = m(x); m.eval(); evaluation = m(x)
    torch.testing.assert_close(train, evaluation, atol=0, rtol=0)


def test_real_optimizer_restore_and_parameter_ownership():
    m = model(config(hca_noise=True)).train()
    opt, _ = optimizer_for(m)
    routing = {id(p) for b in m.blocks for module in (b.attn.pool_score, b.attn.boundary) for p in module.parameters()}
    assert all(g['kind'] == 'adamw' for g in opt.param_groups if any(id(p) in routing for p in g['params']))
    x = torch.randint(0, 64, (1, 32), device=DEVICE)
    def update():
        opt.zero_grad(set_to_none=True)
        loss = m(x, (x+1) % 64, dense=True); loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        opt.step()
        return loss.detach()
    update()
    saved_m, saved_o = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict())
    rng = torch.cuda.get_rng_state()
    loss1 = update(); wanted = copy.deepcopy(m.state_dict())
    m.load_state_dict(saved_m); opt.load_state_dict(saved_o); torch.cuda.set_rng_state(rng)
    loss2 = update()
    torch.testing.assert_close(loss1, loss2, rtol=0, atol=0)
    for key, value in m.state_dict().items():
        torch.testing.assert_close(value, wanted[key], atol=0, rtol=0)


def test_cce_loss_and_gradients():
    m = model(config(), torch.bfloat16).train()
    x = torch.randint(0, 64, (1, 32), device=DEVICE)
    dense = m(x, (x+1) % 64, dense=True)
    dense.backward()
    expected = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    cce = m(x, (x+1) % 64)
    cce.backward()
    torch.testing.assert_close(cce, dense, rtol=.002, atol=.002)
    for name in ('blocks.0.attn.pool_score.weight', 'blocks.0.attn.boundary.weight'):
        g = dict(m.named_parameters())[name].grad
        assert torch.isfinite(g).all() and g.abs().max() > 0
        relative = (g-expected[name]).norm() / expected[name].norm().clamp_min(1e-8)
        assert relative < .05, (name, relative.item())
