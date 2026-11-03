"""Post-attention stream merge: independent oracle, isolation and model tests."""
from dataclasses import asdict
import math

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.engine import KVCache
from nanochat.gpt import CausalSelfAttention, GPTConfig
from nanochat.nvidia_connections import NvidiaHyperConnection
from nanochat.nvidia_ops import nvidia_ops
from nanochat.packed_training import seed_all, snapshot, restore_rng
from nanochat.stream_attention import StreamAttention
from test_connection_training_resume import equal_tree
from test_mhc_nvidia_full import relative, reference_mix, record


@pytest.fixture(autouse=True)
def attention_policy():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    configure_attention('sdpa_cudnn')


def rotary(q, cos, sin):
    a, b = q.chunk(2, dim=-1)
    return torch.cat([a*cos + b*sin, b*cos - a*sin], dim=-1)


def reference_attention(module, x, pre, cos, sin):
    """Literal stream loop and materialized causal attention, independent of bmm."""
    b, t, d, k = x.shape
    outputs = []
    for s in range(k):
        h = F.rms_norm(x[..., s], (d,))
        q, key, value = [F.linear(h, layers[s].weight).reshape(
            b, t, module.stream_heads, module.head_dim) for layers in (module.c_q, module.c_k, module.c_v)]
        q = F.rms_norm(rotary(q, cos, sin), (module.head_dim,)) * 1.2
        key = F.rms_norm(rotary(key, cos, sin), (module.head_dim,)) * 1.2
        score = torch.einsum('bthd,buhd->bhtu', q, key) / math.sqrt(module.head_dim)
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        prob = score.masked_fill(mask, -torch.inf).softmax(-1)
        attended = torch.einsum('bhtu,buhd->bthd', prob, value).reshape(b, t, -1)
        outputs.append(F.linear(attended, module.c_proj[s].weight))
    return sum(y * (module.streams * pre[..., s:s+1]) for s, y in enumerate(outputs))


def gradient_comparison(actual, expected):
    """Retain 2% relative bound, with the existing parity suite's 1e-6 floor.

    RMSNorm cancels common-scale gradients when initial streams nearly agree.
    Report absolute errors/norms, not only ratios with near-zero denominators.
    This test-only comparison does not change model numerics or training.
    """
    stats = {}; squared_error = squared_reference = 0.
    for name, ref in expected.items():
        value = actual[name].double(); ref = ref.double(); delta = value-ref
        error_norm, ref_norm = delta.norm().item(), ref.norm().item()
        bound = 1e-6*math.sqrt(ref.numel()) + .02*ref_norm
        stats[name] = dict(reference_l2=ref_norm, actual_l2=value.norm().item(),
            error_l2=error_norm, max_abs=delta.abs().max().item(),
            relative_l2=error_norm/max(ref_norm, 1e-30), mixed_l2_ratio=error_norm/bound,
            finite=bool(torch.isfinite(value).all() and torch.isfinite(ref).all()))
        squared_error += error_norm**2; squared_reference += ref_norm**2
    global_relative = math.sqrt(squared_error/max(squared_reference, 1e-60))
    passed = global_relative < .02 and all(s['finite'] and s['mixed_l2_ratio'] <= 1 for s in stats.values())
    return dict(passed=passed, global_relative_l2=global_relative, parameters=stats,
        tolerance=dict(atol=1e-6, rtol=.02, global_rtol=.02))


@pytest.mark.parametrize('heads', [4, 8])
def test_grouped_attention_fp64_loop_forward_backward(heads):
    seed_all(102)
    module = StreamAttention(256, heads, 4, 0).cuda().double()
    x = torch.randn(2, 11, 256, 4, device='cuda', dtype=torch.float64, requires_grad=True)
    pre = torch.randn(2, 11, 4, device='cuda', dtype=torch.float64, requires_grad=True)
    angle = torch.randn(1, 11, 1, 128//heads, device='cuda', dtype=torch.float64)
    cos, sin = angle.cos(), angle.sin()
    # FP64 oracle must retain FP64 in the weighted-merge multiplication too.
    with sdpa_kernel(SDPBackend.MATH):
        actual = module(x, pre, (cos, sin))
    expected = reference_attention(module, x, pre, cos, sin)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    probe = torch.randn_like(actual)
    targets = (x, pre, *module.parameters())
    ga = torch.autograd.grad((actual*probe).sum(), targets)
    gr = torch.autograd.grad((expected*probe).sum(), targets)
    for a, r in zip(ga, gr):
        torch.testing.assert_close(a, r, atol=2e-11, rtol=2e-10)
    record(f'stream-loop-h{heads}', dict(output=relative(actual, expected),
        max_gradient_error=max(relative(a, r) for a, r in zip(ga, gr))))


def test_compiled_attention_fp32_loop_forward_backward():
    seed_all(102)
    module = StreamAttention(128, 4, 4, 0).cuda()
    x = torch.randn(2, 17, 128, 4, device='cuda', requires_grad=True)
    pre = torch.randn(2, 17, 4, device='cuda', requires_grad=True)
    angle = torch.randn(1, 17, 1, 16, device='cuda'); cos, sin = angle.cos(), angle.sin()
    with sdpa_kernel(SDPBackend.MATH):
        actual = torch.compile(module.forward, fullgraph=True, dynamic=False)(x, pre, (cos, sin))
        expected = reference_attention(module, x, pre, cos, sin)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=3e-4)
        probe = torch.randn_like(actual)
        ga = torch.autograd.grad((actual*probe).sum(), (x, pre, *module.parameters()))
        gr = torch.autograd.grad((expected*probe).sum(), (x, pre, *module.parameters()))
    errors = dict(output=relative(actual, expected), gradients=[relative(a, r) for a, r in zip(ga, gr)])
    record('stream-compiled-fp32-loop', errors)
    assert max(errors['gradients']) < 3e-4, errors
    for a, r in zip(ga, gr):
        torch.testing.assert_close(a, r, atol=1e-5, rtol=3e-4)


def test_stream_input_and_parameter_isolation_before_merge():
    module = StreamAttention(128, 4, 4, 0).cuda().double()
    x = torch.randn(2, 7, 128, 4, device='cuda', dtype=torch.float64, requires_grad=True)
    pre = torch.zeros(2, 7, 4, device='cuda', dtype=torch.float64); pre[..., 0] = 1
    cos = torch.ones(1, 7, 1, 16, device='cuda', dtype=torch.float64); sin = cos*0
    with sdpa_kernel(SDPBackend.MATH):
        output = module(x, pre, (cos, sin))
        changed = x.detach().clone(); changed[..., 1:] += 10
        torch.testing.assert_close(output, module(changed, pre, (cos, sin)), atol=0, rtol=0)
    output.square().sum().backward()
    assert torch.count_nonzero(x.grad[..., 1:]) == 0 and x.grad[..., 0].norm() > 0
    for layers in (module.c_q, module.c_k, module.c_v, module.c_proj):
        assert layers[0].weight.grad.norm() > 0
        assert all(torch.count_nonzero(p.weight.grad) == 0 for p in layers[1:])


def test_identical_streams_dense_head_partition_and_initial_scale():
    dense = CausalSelfAttention(GPTConfig(n_embd=128, n_head=4, n_kv_head=4, n_layer=1), 0).cuda().double()
    dense.ve_gate = None
    module = StreamAttention(128, 4, 4, 0).cuda().double(); module.copy_from_dense(dense)
    h = torch.randn(2, 13, 128, device='cuda', dtype=torch.float64)
    x = h.unsqueeze(-1).expand(-1, -1, -1, 4)
    cos = torch.ones(1, 13, 1, 16, device='cuda', dtype=torch.float64); sin = cos*0
    pre = torch.full((2, 13, 4), .25, device='cuda', dtype=torch.float64)
    with sdpa_kernel(SDPBackend.MATH):
        base = dense(F.rms_norm(h, (128,)), None, (cos, sin), (-1, 0), None)
        actual = module(x, pre, (cos, sin))
        uncompensated = module(x, pre/4, (cos, sin))
    torch.testing.assert_close(actual, base, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(uncompensated, base/4, atol=2e-12, rtol=2e-12)
    probe = torch.randn_like(base)
    base_params = [getattr(dense, name).weight for name in ('c_q', 'c_k', 'c_v', 'c_proj')]
    base_grads = torch.autograd.grad((base*probe).sum(), base_params)
    split_grads = torch.autograd.grad((actual*probe).sum(), tuple(module.parameters()))
    for i, name in enumerate(('c_q', 'c_k', 'c_v', 'c_proj')):
        joined = torch.cat(split_grads[4*i:4*(i+1)], dim=1 if name=='c_proj' else 0)
        torch.testing.assert_close(joined, base_grads[i], atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize('sinkhorn', [True, False])
def test_unaggregated_nvidia_routing_forward_backward(sinkhorn):
    seed_all(45)
    layer = NvidiaHyperConnection(128, 4, 0, sinkhorn=sinkhorn).cuda(); layer.reset_parameters()
    x = torch.randn(2, 64, 128, 4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    pre, skip, mix = layer.routing(x)
    y = torch.sin(x.float()).to(x.dtype)
    merged = nvidia_ops().mhc_fused_aggregate(y, pre, 4, use_tf32=True, fused_grad_x_acc_buffer=None)
    actual = layer.write(merged, skip, mix)
    xr = x.detach().double().requires_grad_()
    params = [p.detach().double().requires_grad_() for p in layer.parameters()]
    phi, bias, scales = params
    pr, po, comb = reference_mix(xr, phi, scales, bias)
    if not sinkhorn:
        flat = xr.flatten(-2)
        raw = F.linear(flat/(flat.square().mean(-1, keepdim=True)+torch.finfo(torch.float32).eps).sqrt(), phi)
        comb = (raw[..., 8:]*scales[2]+bias[8:]).reshape(2, 64, 4, 4)
    br = (xr.sin()*pr.unsqueeze(-2)).sum(-1)
    expected = torch.einsum('btcn,btnm->btcm', xr, comb) + br.unsqueeze(-1)*po.unsqueeze(-2)
    probe = torch.randn_like(actual).float()/actual.numel()
    ga = torch.autograd.grad((actual.float()*probe).sum(), (x, *layer.parameters()))
    gr = torch.autograd.grad((expected*probe).sum(), (xr, *params))
    errors = dict(output=relative(actual, expected), gradients=[relative(a, b) for a, b in zip(ga, gr)])
    record(f'stream-routing-sinkhorn{sinkhorn}', errors)
    assert errors['output'] < .01 and max(errors['gradients']) < .03, errors


def build(mode='per_stream', compiled=False, length=2048, depth=2):
    seed_all(42)
    config = ConnectionConfig(variant='mhc', attention_mode=mode, width=512, heads=4,
        depth=depth, sequence_length=length, vocab_size=256)
    model = ConnectionModel(config).prepare(compile_blocks=compiled)
    match_vanilla_initialization(model, 42)
    return model


def test_parameter_budget_matching_optimizer_and_config_compatibility():
    shared = build('shared', length=32); separate = build(length=32)
    assert sum(p.numel() for p in shared.parameters()) == sum(p.numel() for p in separate.parameters())
    for name, p in shared.named_parameters():
        if '.attn.' not in name:
            assert torch.equal(p, separate.get_parameter(name)), name
    for old, new in zip(shared.blocks, separate.blocks):
        for name in ('c_q', 'c_k', 'c_v', 'c_proj'):
            axis = 1 if name == 'c_proj' else 0
            assert torch.equal(getattr(old.attn, name).weight,
                               torch.cat([p.weight for p in getattr(new.attn, name)], dim=axis))
    opt, _ = optimizer_for(separate)
    muon_ids = {id(p) for g in opt.param_groups if g['kind']=='muon' for p in g['params']}
    assert all(id(p) in muon_ids and p.ndim==2 for b in separate.blocks for p in b.attn.parameters())
    legacy = asdict(shared.config); legacy.pop('attention_mode')
    assert ConnectionConfig.from_checkpoint(legacy).attention_mode == 'shared'
    assert ConnectionConfig.from_checkpoint(asdict(separate.config)) == separate.config
    for kwargs in (dict(variant='baseline'), dict(variant='mhc', heads=2),
                   dict(variant='mhc', architecture='native'), dict(variant='mhc', fused_connection_grad=True)):
        with pytest.raises(ValueError): ConnectionConfig(attention_mode='per_stream', **kwargs)


@pytest.mark.parametrize('compiled', [False, True])
def test_model_causality_cache_loss_gradients_recompute(compiled):
    configure_attention('sdpa_cudnn')
    model = build(compiled=compiled)
    with torch.no_grad():
        for block in model.blocks:
            for p in block.attn.c_proj: p.weight.normal_(std=.015)
            block.mlp.c_proj.weight.normal_(std=.015)
    ids = torch.randint(0, 256, (2, 2048), device='cuda')
    changed = ids.clone(); changed[0, 1024:] = (changed[0, 1024:]+19)%256
    with torch.no_grad():
        a, b = model(ids), model(changed)
        assert torch.equal(a[0, :1024], b[0, :1024])
        assert torch.equal(a[1], b[1])
        # Cache is an inference path, not a compiled training path.
        for block in model.blocks:
            block.compiled_attention = None; block.compiled_mlp = None
        cache = KVCache(2, 4, 32, 128, 2, 'cuda', torch.bfloat16)
        pre = model(ids[:, :7], kv_cache=cache)
        one = model(ids[:, 7:8], kv_cache=cache); full = model(ids[:, :8])
        torch.testing.assert_close(pre, full[:, :7], atol=.004, rtol=.03)
        torch.testing.assert_close(one, full[:, 7:], atol=.004, rtol=.03)
        assert cache.get_pos() == 8
    if compiled: model.prepare(compile_blocks=True)
    x, y = ids[:, :128].contiguous(), ids[:, 1:129].contiguous()
    loss = model(x, y); dense = model(x, y, dense=True)
    torch.testing.assert_close(loss, dense, atol=2e-4, rtol=2e-4)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in model.parameters())
    before = {k:p.grad.clone() for k,p in model.named_parameters()}
    model.zero_grad(set_to_none=True); model.config.activation_checkpoint = True
    again = model(x, y); again.backward()
    torch.testing.assert_close(loss, again, atol=2e-5, rtol=2e-5)
    errors = {k:relative(p.grad, before[k]) for k,p in model.named_parameters()}
    record(f'stream-model-compiled{compiled}', dict(recompute_gradient_errors=errors))
    assert max(errors.values()) < .02, errors
    if compiled:
        model.config.activation_checkpoint = False
        for block in model.blocks:
            block.compiled_attention = None; block.compiled_mlp = None
        model.zero_grad(set_to_none=True)
        eager = model(x, y); eager.backward()
        torch.testing.assert_close(loss, eager, atol=2e-4, rtol=2e-4)
        stats = gradient_comparison({k:p.grad for k,p in model.named_parameters()}, before)
        record('stream-eager-compiled', dict(loss_compiled=loss.item(), loss_eager=eager.item(), **stats))
        assert stats['passed'], stats


def test_shared_mhc_eager_compiled_numerical_control():
    model = build(mode='shared', compiled=True)
    with torch.no_grad():
        for block in model.blocks:
            block.attn.c_proj.weight.normal_(std=.015)
            block.mlp.c_proj.weight.normal_(std=.015)
    ids = torch.randint(0, 256, (2, 2048), device='cuda')
    x, y = ids[:, :128].contiguous(), ids[:, 1:129].contiguous()
    compiled = model(x, y); compiled.backward()
    before = {k:p.grad.clone() for k,p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    repeat = model(x, y); repeat.backward()
    repeated = gradient_comparison({k:p.grad for k,p in model.named_parameters()}, before)
    for block in model.blocks:
        block.compiled_attention = None; block.compiled_mlp = None
    model.zero_grad(set_to_none=True)
    eager = model(x, y); eager.backward()
    stats = gradient_comparison({k:p.grad for k,p in model.named_parameters()}, before)
    record('shared-eager-compiled-control', dict(loss_compiled=compiled.item(), loss_eager=eager.item(),
        repeat_control=repeated, **stats))
    torch.testing.assert_close(compiled, eager, atol=2e-4, rtol=2e-4)
    assert stats['passed'] and repeated['passed'], stats


def test_optimizer_checkpoint_fixed_gradient_next_update(tmp_path):
    configure_attention('sdpa_cudnn')
    model = build(length=128); opt, wd = optimizer_for(model)
    x = torch.randint(0, 256, (2, 128), device='cuda'); y = (x+1)%256
    for step in range(4):
        schedule(opt, step, 305176, wd); opt.zero_grad(set_to_none=True)
        loss = model(x, y); loss.backward(); opt.step()
        assert torch.isfinite(loss)
    gradients = {k:p.grad.detach().clone() for k,p in model.named_parameters()}
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in gradients.values())
    path = tmp_path/'checkpoint.pt'
    torch.save(snapshot(model, opt, 4, config=asdict(model.config)), path)
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    other = build(length=128); other_opt, _ = optimizer_for(other)
    other.load_state_dict(loaded['model']); other_opt.load_state_dict(loaded['optimizer'])
    equal_tree(model.state_dict(), other.state_dict()); equal_tree(opt.state_dict(), other_opt.state_dict())
    restore_rng(loaded['rng']); expected = torch.rand(8, device='cuda')
    restore_rng(loaded['rng']); assert torch.equal(expected, torch.rand(8, device='cuda'))
    for target, optimizer in ((model, opt), (other, other_opt)):
        for name, p in target.named_parameters(): p.grad = gradients[name].clone()
        schedule(optimizer, 4, 305176, wd); optimizer.step()
    equal_tree(model.state_dict(), other.state_dict()); equal_tree(opt.state_dict(), other_opt.state_dict())
