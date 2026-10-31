"""Native-backbone preservation and real NVIDIA connection integration."""
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import MethodType

import numpy as np
import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
from nanochat.engine import KVCache
from nanochat.nvidia_connections import expand_streams
from nanochat.nvidia_ops import nvidia_ops
from nanochat.packed_training import seed_all, snapshot, restore_rng
from test_connection_training_resume import equal_tree


def record(name, value):
    root = Path('results/native-checks')
    root.mkdir(parents=True, exist_ok=True)
    (root/(name+'.json')).write_text(json.dumps(value, indent=2)+'\n')


def native(variant='baseline', **kwargs):
    config = dict(n_layer=4, n_embd=128, n_head=2, n_kv_head=2,
                  vocab_size=256, sequence_len=2048, window_pattern='SSSL')
    config.update(kwargs)
    seed_all(42)
    with torch.device('meta'):
        model = GPT(GPTConfig(connection_variant=variant, **config))
    model.to_empty(device='cuda').init_weights()
    return model


def exercise_features(model):
    # Original zero projections/smear would hide omitted VE, smear or attention.
    with torch.no_grad():
        model.smear_lambda.fill_(.3)
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=.03)
            block.mlp.c_proj.weight.normal_(std=.03)


def optimizer_signature(model, opt):
    names = {id(p): n for n, p in model.named_parameters()}
    return [dict(**{k: v for k, v in g.items() if k != 'params'},
                 params=[names[id(p)] for p in g['params']]) for g in opt.param_groups]


def test_original_native_baseline_parity():
    # Immutable pre-change GPT, not a second invocation of the new code.
    spec = importlib.util.spec_from_file_location('native_gpt_before_connections', os.environ['NATIVE_REFERENCE_GPT'])
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    options = dict(n_layer=4, n_embd=128, n_head=2, n_kv_head=2,
                   vocab_size=256, sequence_len=2048, window_pattern='SSSL')
    seed_all(42)
    with torch.device('meta'):
        reference = old.GPT(old.GPTConfig(**options))
    reference.to_empty(device='cuda').init_weights()
    model = native(**options)
    equal_tree(reference.state_dict(), model.state_dict())
    exercise_features(reference)
    model.load_state_dict(reference.state_dict())
    ids = torch.randint(0, 256, (1, 2048), device='cuda')
    labels = (ids+3) % 256
    a, b = reference(ids, labels), model(ids, labels)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.backward(); b.backward()
    for (name, p), (other_name, q) in zip(reference.named_parameters(), model.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=3e-4)
    assert reference.num_scaling_params() == model.num_scaling_params()
    assert optimizer_signature(reference, reference.setup_optimizer()) == optimizer_signature(model, model.setup_optimizer())
    record('native-parity', dict(loss=a.item(), native_parameters_equal=True,
        windows=model.window_sizes, optimizer_groups_equal=True, causal_length=2048))


def test_full_size_backbone_initialization_and_optimizer_groups():
    models = [native(v, n_layer=8, n_embd=512, n_head=4, n_kv_head=4, vocab_size=32768)
              for v in ('baseline', 'mhc', 'mhc_no_sinkhorn')]
    a, b, c = [m.state_dict() for m in models]
    assert all(torch.equal(value, b[name]) and torch.equal(value, c[name]) for name, value in a.items())
    equal_tree(b, c)
    signatures = [optimizer_signature(m, m.setup_optimizer()) for m in models]
    assert signatures[1] == signatures[2]
    base_names = set(a)
    for sig in signatures[1:]:
        original = [g for g in sig if any(p in base_names for p in g['params'])]
        assert original == signatures[0]
        connector = [g for g in sig if not any(p in base_names for p in g['params'])]
        assert len(connector) == 1 and connector[0]['kind'] == 'adamw' and connector[0]['weight_decay'] == 0
    counts = [m.num_scaling_params() for m in models]
    assert all(count['total'] == sum(p.numel() for p in m.parameters()) for m, count in zip(models, counts))
    from torch._dynamo import config as compiler_config
    params = dict(models[1].named_parameters())
    shapes = {(tuple(params[n].shape), str(params[n].dtype))
              for g in signatures[1] if g['kind'] == 'adamw' for n in g['params']}
    assert len(shapes) > 8 and compiler_config.recompile_limit >= len(shapes)
    record('initialization', dict(counts=counts, all_shared_parameters_bitwise_equal=True,
        mhc_and_no_sinkhorn_all_parameters_equal=True, native_optimizer_groups_unchanged=True,
        adamw_shape_dtype_variants=len(shapes), compiler_recompile_limit=compiler_config.recompile_limit))


def test_streamwise_native_extras_reduce_to_original_model():
    reference = native()
    exercise_features(reference)
    model = native('mhc')
    missing, extra = model.load_state_dict(reference.state_dict(), strict=False)
    assert not extra and all('_connection.' in n or n.startswith('stream_head.') for n in missing)
    # Replace only the connection operation by exact ordinary residual updates.
    # Native forward must then reduce to GPT despite its additional stream axis.
    for block in model.transformer.h:
        for connection in (block.attn_connection, block.mlp_connection):
            connection.read = MethodType(lambda self, x: (x[..., 0], x, None), connection)
            connection.write = MethodType(lambda self, y, x, mixes: expand_streams(x[..., 0]+y), connection)
    model.stream_head.forward = MethodType(lambda self, x: x[..., 0], model.stream_head)
    ids = torch.randint(0, 256, (1, 128), device='cuda')
    a, b = reference(ids, (ids+1)%256), model(ids, (ids+1)%256)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.backward(); b.backward()
    new_params = dict(model.named_parameters())
    for name, p in reference.named_parameters():
        torch.testing.assert_close(p.grad, new_params[name].grad, atol=1e-6, rtol=3e-4)
    record('streamwise-extras', dict(loss=a.item(), native_extras_and_gradients_preserved=True))


@pytest.mark.parametrize('variant', ['mhc', 'mhc_no_sinkhorn'])
def test_real_kernels_causal_swa_gqa_cache_and_feature_gradients(variant, monkeypatch):
    if variant == 'mhc_no_sinkhorn':
        def forbidden(*args, **kwargs):
            raise AssertionError('No-Sinkhorn executed manifold projection')
        monkeypatch.setattr(nvidia_ops(), 'mhc_generate_mix_and_aggregate', forbidden)
        monkeypatch.setattr(nvidia_ops(), 'mhc_fused_sinkhorn', forbidden)
    model = native(variant, n_kv_head=1)
    exercise_features(model)
    assert model.window_sizes == [(512, 0)]*3 + [(2048, 0)]
    assert list(model.value_embeds) == ['1', '3']
    ids = torch.randint(0, 256, (1, 2048), device='cuda')
    changed = ids.clone(); changed[:, 1024:] = (changed[:, 1024:]+17)%256
    with torch.no_grad():
        a, b = model(ids), model(changed)
        torch.testing.assert_close(a[:, :1024], b[:, :1024], atol=0, rtol=0)
        # Cross the SWA boundary; smear must also survive one-token decoding.
        cache = KVCache(1, 1, 600, 64, 4, 'cuda', torch.bfloat16)
        pre = model(ids[:, :520], kv_cache=cache)
        one = model(ids[:, 520:521], kv_cache=cache)
        full = model(ids[:, :521])
        torch.testing.assert_close(pre, full[:, :520], atol=.004, rtol=.03)
        torch.testing.assert_close(one, full[:, 520:], atol=.004, rtol=.03)
    loss = model(ids[:, :128], (ids[:, :128]+1)%256)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    features = {n: p.grad.float().norm().item() for n, p in model.named_parameters()
                if n in ('resid_lambdas', 'x0_lambdas', 'smear_lambda', 'backout_lambda', 'smear_gate.weight')
                or 'value_embeds.' in n or 've_gate.' in n}
    assert features and all(v > 0 for v in features.values()), features
    record(variant+'-integration', dict(causal_tokens=2048, cache_across_swa_boundary=True,
        gqa=True, feature_gradient_norms=features, loss=loss.item()))


def adapter(variant, compile_blocks=False):
    seed_all(42)
    return build_connection_model(ConnectionConfig(architecture='native', variant=variant,
        depth=4, width=128, heads=2, vocab_size=256, sequence_length=2048)).prepare(compile_blocks=compile_blocks)


@pytest.mark.parametrize('variant', ['baseline', 'mhc', 'mhc_no_sinkhorn'])
def test_cce_compile_checkpoint_and_resume(variant, tmp_path):
    model = adapter(variant)
    exercise_features(model.native)
    ids = torch.randint(0, 256, (1, 128), device='cuda'); targets = (ids+2)%256
    dense = model(ids, targets, dense=True)
    loss = model(ids, targets)
    torch.testing.assert_close(loss, dense, atol=2e-4, rtol=2e-4)
    loss.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    model.config.activation_checkpoint = True
    again = model(ids, targets); again.backward()
    torch.testing.assert_close(again, loss, atol=2e-5, rtol=2e-5)
    # BF16 accumulation can differ; compare full-vector error and absolute floor.
    for name, p in model.named_parameters():
        error = (p.grad.float()-gradients[name].float()).norm()
        assert error <= .02*gradients[name].float().norm()+1e-6, name
    model.config.activation_checkpoint = False
    model.native.compile_branches()
    opt, wd = optimizer_for(model)
    for step in range(2):
        schedule(opt, step, 305176, wd)
        opt.zero_grad(set_to_none=True)
        compiled_loss = model(ids, targets); compiled_loss.backward(); opt.step()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
    path = tmp_path/'latest.pt'
    torch.save(snapshot(model, opt, 2, config=asdict(model.config), next_data_row=32), path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    other = adapter(variant); other_opt, other_wd = optimizer_for(other)
    other.load_state_dict(state['model']); other_opt.load_state_dict(state['optimizer'])
    assert state['step'] == 2 and state['next_data_row'] == 32 and other_wd == wd
    assert ConnectionConfig.from_checkpoint(state['config']) == other.config
    equal_tree(model.state_dict(), other.state_dict()); equal_tree(opt.state_dict(), other_opt.state_dict())
    restore_rng(state['rng']); expected = (torch.rand(5), torch.rand(5, device='cuda'), np.random.rand(5))
    restore_rng(state['rng']); actual = (torch.rand(5), torch.rand(5, device='cuda'), np.random.rand(5))
    equal_tree(expected, actual)
    for m, o in ((model, opt), (other, other_opt)):
        for name, p in m.named_parameters(): p.grad = gradients[name].clone()
        schedule(o, 2, 305176, wd); o.step()
    equal_tree(model.state_dict(), other.state_dict()); equal_tree(opt.state_dict(), other_opt.state_dict())
    record(variant+'-resume', dict(dense_loss=dense.item(), cce_loss=loss.item(),
        compiled_loss=compiled_loss.item(), exact_fixed_gradient_resume=True,
        activation_checkpoint=True, optimizer_groups=optimizer_signature(model, opt)))
