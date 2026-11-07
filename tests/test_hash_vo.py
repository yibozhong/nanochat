"""GPU acceptance checks for the new fixed-hash VO path, with active branches."""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.engine import KVCache
from nanochat.hash_qkv import balanced_lookup, make_plan, _gather_op
from nanochat.hash_vo import HashVOModel, HashVOProjection
from nanochat.packed_training import seed_all, snapshot, restore_rng
from test_connection_training_resume import equal_tree


@pytest.fixture(autouse=True)
def environment():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    configure_attention('fa3')
    torch._dynamo.reset()
    seed_all(42)
    yield
    torch._dynamo.reset()


def close(actual, reference, tolerance=.02, floor=1e-7):
    a, r = actual.float(), reference.float()
    assert torch.isfinite(a).all() and torch.isfinite(r).all()
    error, scale = (a - r).norm().item(), r.norm().item()
    assert error <= tolerance * scale + floor, (error, scale, error / max(scale, floor))
    return error / max(scale, floor)


def build(*, width=128, length=128, vocab=256, depth=2, compile=False, recompute=False,
          buckets=16, backend='deepgemm', activate=True):
    seed_all(42)
    cfg = ConnectionConfig(width=width, heads=width // 128, sequence_length=length,
        vocab_size=vocab, depth=depth, activation_checkpoint=recompute)
    lookup = balanced_lookup(np.arange(1, vocab + 1), buckets)
    model = HashVOModel(cfg, lookup, buckets=buckets, backend=backend).prepare(compile_blocks=compile)
    if activate:
        with torch.no_grad():
            for block in model.blocks:
                for p in block.attn.o.weights: p.normal_(std=.02)
                block.mlp.c_proj.weight.normal_(std=.02)
    return model


def test_balanced_frequency_assignment_and_unseen_tokens():
    frequencies = np.array([700, 300, 200, 199, 101, 100, 50, 50, 1, 0, 0, 0])
    table = balanced_lookup(frequencies, 4)
    assert np.array_equal(table, balanced_lookup(frequencies, 4))
    masses = np.bincount(table, weights=frequencies, minlength=4)
    assert masses.max() >= frequencies.max()  # no impossible perfect-balance claim
    assert masses.max() - masses.min() <= frequencies.max()
    flat = balanced_lookup(np.ones(32768, dtype=np.int64), 16)
    assert np.array_equal(np.bincount(flat), np.full(16, 2048))
    assert set(balanced_lookup(np.zeros(64, dtype=np.int64), 16)) == set(range(16))


def test_routing_permutation_padding_and_backward():
    table = np.arange(256) % 16
    for ids in (np.arange(257) % 256, np.zeros(257, dtype=np.int64)):
        plan = make_plan(ids, table)
        x = torch.randn(257, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        packed = _gather_op(x, plan.pack, plan.unpack)
        result = _gather_op(packed, plan.unpack, plan.pack)
        assert torch.equal(result, x)
        assert torch.count_nonzero(packed[plan.pack < 0]) == 0
        dy = torch.randn_like(x)
        (dx,) = torch.autograd.grad(result, x, dy)
        assert torch.equal(dx, dy)
        assert plan.pack.numel() == 2304
        assert int(plan.sizes_gpu.sum()) == plan.pack.numel()


@pytest.mark.parametrize('n,width,magnitude,used', [
    (257, 128, 1., 16), (19, 128, 1., 3), (129, 128, 0., 1),
    (129, 128, 1e-4, 16), (257, 512, 1., 16)])
def test_all_projection_vjps_against_fp64(n, width, magnitude, used):
    table = np.arange(256) % 16
    ids = np.arange(n) % used
    plan = make_plan(ids, table)
    original = torch.randn(width, width) / width**.5
    projection = HashVOProjection(original).cuda()
    x = (torch.randn(n, width, device='cuda') * magnitude).bfloat16().requires_grad_()
    dy = torch.randn(n, width, device='cuda', dtype=torch.bfloat16)
    y = projection(x, plan)
    gradients = torch.autograd.grad(y, [x, *projection.parameters()], dy)
    expected_y, expected_dx = torch.empty_like(y), torch.empty_like(x)
    errors = []
    # Direct independent per-expert FP64 matmuls, with the actual BF16 operands.
    for e in range(16):
        rows = torch.as_tensor(np.flatnonzero(ids == e), device='cuda')
        xx, dd = x.detach()[rows].double(), dy[rows].double()
        w = projection.weights[e].detach().bfloat16().double()
        expected_y[rows] = (xx @ w.T).bfloat16()
        expected_dx[rows] = (dd @ w).bfloat16()
        dw = dd.T @ xx
        errors.append(close(gradients[1 + e], dw))
    errors += [close(y, expected_y), close(gradients[0], expected_dx)]
    root = Path('results/hash-vo/checks'); root.mkdir(parents=True, exist_ok=True)
    (root/f'vjp-{n}-{width}-{magnitude}.json').write_text(json.dumps(dict(max_relative_l2=max(errors)))+'\n')


def test_single_bucket_reduces_to_dense_backbone():
    model = build(buckets=1)
    seed_all(42)
    reference = ConnectionModel(model.config).prepare()
    with torch.no_grad():
        reference.wte.weight.copy_(model.wte.weight)
        reference.lm_head.weight.copy_(model.lm_head.weight)
        for a, b in zip(model.blocks, reference.blocks):
            b.mlp.load_state_dict(a.mlp.state_dict())
            b.attn.c_q.load_state_dict(a.attn.c_q.state_dict())
            b.attn.c_k.load_state_dict(a.attn.c_k.state_dict())
            b.attn.c_v.weight.copy_(a.attn.v.weights[0])
            b.attn.c_proj.weight.copy_(a.attn.o.weights[0])
    ids = torch.randint(0, 256, (2, 128), device='cuda')
    close(model(ids), reference(ids), .01)
    a, b = model(ids, (ids + 1) % 256), reference(ids, (ids + 1) % 256)
    assert abs(a.item() - b.item()) < 2e-4
    a.backward(); b.backward()
    for x, y in zip(model.blocks, reference.blocks):
        for name in ('c_q', 'c_k'):
            close(getattr(x.attn,name).weight.grad, getattr(y.attn,name).weight.grad, .03)
        close(x.attn.v.weights[0].grad,y.attn.c_v.weight.grad,.03)
        close(x.attn.o.weights[0].grad,y.attn.c_proj.weight.grad,.03)


def test_full_model_compile_reference_and_changing_routing():
    model = build(compile=True)
    reference = build(backend='reference')
    reference.load_state_dict(model.state_dict())
    for offset in (0, 31):
        ids = (torch.arange(256, device='cuda').reshape(2, 128) + offset) % 256
        # Exercise a different bucket distribution with exactly the same shapes.
        if offset: ids[0].fill_(5)
        targets = (ids + 7) % 256
        model.zero_grad(set_to_none=True); reference.zero_grad(set_to_none=True)
        a, b = model(ids, targets), reference(ids, targets)
        assert abs(a.item() - b.item()) < 3e-4
        a.backward(); b.backward()
        for (name, p), (other, q) in zip(model.named_parameters(), reference.named_parameters()):
            assert name == other
            close(p.grad, q.grad, .03)


def test_causal_2k_batch_isolation_and_kv_cache():
    model = build(width=512, length=2048, vocab=512, depth=2)
    ids = torch.randint(0, 512, (2, 2048), device='cuda')
    changed = ids.clone(); changed[0, 1024:] = (changed[0, 1024:] + 19) % 512
    model.eval()
    with torch.no_grad():
        a, b = model.hidden(ids), model.hidden(changed)
        assert torch.equal(a[0, :1024], b[0, :1024])
        assert torch.equal(a[1], b[1])
        cache = KVCache(2, 4, 2048, 128, 2, 'cuda', torch.bfloat16)
        outputs = [model.hidden(ids[:, :16], cache), model.hidden(ids[:, 16:32], cache)]
        close(torch.cat(outputs, dim=1), model.hidden(ids[:, :32]), .01)
        assert cache.get_pos() == 32


def test_cce_and_recomputation():
    model = build(compile=True)
    reference = build(compile=True, recompute=True)
    reference.load_state_dict(model.state_dict())
    ids = torch.randint(0, 256, (2, 128), device='cuda'); target = (ids + 3) % 256
    a = model(ids, target); b = reference(ids, target, dense=True)
    assert abs(a.item() - b.item()) < 2e-5
    a.backward(); b.backward()
    for p, q in zip(model.parameters(), reference.parameters()):
        close(p.grad, q.grad, .03)


def test_optimizer_checkpoint_rng_and_independent_2d_updates(tmp_path):
    model = build()
    opt, wd = optimizer_for(model)
    vo_ids = {id(p) for b in model.blocks for proj in (b.attn.v,b.attn.o) for p in proj.parameters()}
    for group in opt.param_groups:
        if any(id(p) in vo_ids for p in group['params']):
            assert group['kind'] == 'muon'
            assert all(p.ndim == 2 for p in group['params'])
    ids = torch.randint(0, 256, (2, 128), device='cuda'); target = (ids + 1) % 256
    for step in range(4):
        schedule(opt, step, 305176, wd); opt.zero_grad(set_to_none=True)
        model(ids, target).backward(); opt.step()
    grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    state = snapshot(model, opt, 4, config=asdict(model.config), hash_method='frequency_balanced', buckets=16)
    path = tmp_path/'checkpoint.pt'; torch.save(state, path)
    saved = torch.load(path, weights_only=False, map_location='cpu')
    other = build(); other_opt, other_wd = optimizer_for(other)
    other.load_state_dict(saved['model']); other_opt.load_state_dict(saved['optimizer'])
    equal_tree(model.state_dict(), other.state_dict()); equal_tree(opt.state_dict(), other_opt.state_dict())
    assert np.array_equal(other._lookup_cpu, model._lookup_cpu)
    restore_rng(state['rng']); expected = (torch.rand(8), torch.rand(8, device='cuda'), np.random.rand(8))
    restore_rng(saved['rng']); actual = (torch.rand(8), torch.rand(8, device='cuda'), np.random.rand(8))
    equal_tree(expected, actual)
    for obj, optimizer in ((model, opt), (other, other_opt)):
        for n, p in obj.named_parameters(): p.grad = grads[n].clone()
        schedule(optimizer, 4, 305176, wd); optimizer.step()
    equal_tree(model.state_dict(), other.state_dict()); equal_tree(opt.state_dict(), other_opt.state_dict())
    bad = dict(saved['model']); bad['hash_buckets'] = torch.tensor(8)
    with pytest.raises(ValueError): other.load_state_dict(bad)


def test_shared_qk_initialization_and_zero_output_experts():
    seed_all(42)
    cfg = ConnectionConfig(width=128, heads=1, sequence_length=128, vocab_size=256, depth=2)
    dense = ConnectionModel(cfg)
    expected_rng = torch.get_rng_state().clone()
    seed_all(42)
    model = HashVOModel(cfg, np.arange(256) % 16)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in dense.parameters()) + 2*15*cfg.depth*cfg.width**2
    assert 58720256 + 2*15*8*512**2 == 121634816
    for a, b in zip(model.blocks, dense.blocks):
        assert torch.equal(a.attn.c_q.weight, b.attn.c_q.weight)
        assert torch.equal(a.attn.c_k.weight, b.attn.c_k.weight)
        assert torch.equal(a.attn.v.weights[0], b.attn.c_v.weight)
        assert not torch.equal(a.attn.v.weights[0], a.attn.v.weights[1])
        assert all(torch.count_nonzero(p) == 0 for p in a.attn.o.weights)
        equal_tree(a.mlp.state_dict(), b.mlp.state_dict())
    assert torch.equal(model.wte.weight, dense.wte.weight)
    assert torch.equal(model.lm_head.weight, dense.lm_head.weight)


def test_source_value_and_destination_output_routing_independent_oracle():
    from nanochat.gpt import apply_rotary_emb, norm
    from nanochat.flash_attention import flash_attn
    from torch.nn import functional as F
    model = build()
    attn = model.blocks[0].attn
    ids = torch.arange(38).reshape(2, 19) % 256
    assignment = model._lookup_cpu[ids.numpy().reshape(-1)].tolist()
    plan = model.make_routing_plan(ids)
    x = torch.randn(2, 19, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    cos, sin = model.cos[:, :19], model.sin[:, :19]
    actual = attn(x, cos, sin, plan)
    # Ordinary token-by-token linear operations never consume the routing plan.
    flat = x.reshape(-1, 128)
    v = torch.stack([F.linear(row, attn.v.weights[e].bfloat16()) for row, e in zip(flat, assignment)])
    q = attn.c_q(x).reshape(2, 19, 1, 128)
    k = attn.c_k(x).reshape(2, 19, 1, 128)
    q = norm(apply_rotary_emb(q, cos, sin)) * 1.2
    k = norm(apply_rotary_emb(k, cos, sin)) * 1.2
    mixed = flash_attn.flash_attn_func(q, k, v.reshape(2, 19, 1, 128).contiguous(),
                                      causal=True, window_size=(-1, 0)).reshape(-1, 128)
    expected = torch.stack([F.linear(row, attn.o.weights[e].bfloat16()) for row, e in zip(mixed, assignment)]).reshape_as(x)
    close(actual, expected, .02)
    dy = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, [x, *attn.parameters()], dy)
    gb = torch.autograd.grad(expected, [x, *attn.parameters()], dy, allow_unused=True)
    for a, b in zip(ga, gb):
        close(a, torch.zeros_like(a) if b is None else b, .03)


def test_qkv_checkpoint_is_rejected_even_without_strict_loading():
    model = build(activate=False)
    old = {k: v for k, v in model.state_dict().items() if k != 'hash_projection_kind'}
    with pytest.raises(ValueError):
        model.load_state_dict(old, strict=False)
