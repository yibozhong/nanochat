"""Exact native LLSSSSSS layout, preserving native features and checkpoint behavior."""
from dataclasses import asdict
import gc

import pytest
import torch

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for
from nanochat.engine import KVCache
from nanochat.packed_training import seed_all
import test_native_connections as original
from test_connection_training_resume import equal_tree


@pytest.fixture(autouse=True)
def gpu():
    assert torch.cuda.get_device_capability() == (9, 0)
    assert configure_attention('hardware_auto') == 'fa3'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()


def adapter(**kwargs):
    seed_all(42)
    cfg = dict(architecture='native', variant='baseline', streams=1, depth=8,
               width=128, heads=2, vocab_size=256, sequence_length=2048,
               window_pattern='LLSSSSSS', window_final_full=False)
    cfg.update(kwargs)
    return build_connection_model(ConnectionConfig(**cfg)).prepare()


def test_default_matches_immutable_native():
    original.test_original_native_baseline_parity()


def test_production_initialization_parameters_optimizer_and_layout():
    a = adapter(width=512, heads=4, vocab_size=32768)
    rng = torch.cuda.get_rng_state()
    b = adapter(width=512, heads=4, vocab_size=32768, window_pattern='SSSL', window_final_full=True)
    assert torch.equal(rng, torch.cuda.get_rng_state())
    equal_tree(a.state_dict(), b.state_dict())
    assert sum(p.numel() for p in a.parameters()) == 125829354
    assert a.native.window_sizes == [(2048,0)]*2+[(512,0)]*6
    assert b.native.window_sizes == ([(512,0)]*3+[(2048,0)])*2
    assert sorted(a.native.window_sizes) == sorted(b.native.window_sizes)
    oa, wa = optimizer_for(a); ob, wb = optimizer_for(b)
    assert wa == wb
    assert original.optimizer_signature(a,oa) == original.optimizer_signature(b,ob)
    assert a.native.stream_head is None


def test_checkpoint_metadata_and_default_override():
    cfg = ConnectionConfig(architecture='native', window_pattern='LLSSSSSS', window_final_full=False)
    assert ConnectionConfig.from_checkpoint(asdict(cfg)) == cfg
    old = asdict(cfg); old.pop('window_final_full')
    assert ConnectionConfig.from_checkpoint(old).window_final_full is True
    m = adapter(window_final_full=True)
    assert m.native.window_sizes == [(2048,0)]*2+[(512,0)]*5+[(2048,0)]
    with pytest.raises(ValueError):
        ConnectionConfig(window_final_full=False)
    with pytest.raises(ValueError):
        ConnectionConfig(architecture='native',window_final_full='false')


@pytest.mark.parametrize('layer', [0, 1, 2, 7])
def test_real_attention_layer_window(layer):
    m = adapter().native
    original.exercise_features(m)
    x = torch.randn(1,768,128,device='cuda',dtype=torch.bfloat16)
    changed = x.clone(); changed[:,:128] = torch.randn_like(changed[:,:128])
    attn = m.transformer.h[layer].attn
    cs = (m.cos[:,:768],m.sin[:,:768])
    with torch.no_grad():
        a = attn(x,None,cs,m.window_sizes[layer],None)
        b = attn(changed,None,cs,m.window_sizes[layer],None)
    if layer >= 2:
        torch.testing.assert_close(a[:,-1],b[:,-1],atol=0,rtol=0)
    else:
        assert not torch.equal(a[:,-1],b[:,-1])


def test_2k_causality_batch_independence_and_native_feature_gradients():
    m = adapter(); original.exercise_features(m.native)
    ids = torch.randint(0,256,(2,2048),device='cuda')
    changed = ids.clone(); changed[0,1024:] = (changed[0,1024:]+17)%256
    with torch.no_grad():
        a,b = m.hidden(ids),m.hidden(changed)
        assert torch.equal(a[0,:1024],b[0,:1024])
        assert torch.equal(a[1],b[1])
    x = ids[:1,:128]
    m(x,(x+1)%256).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    features = [p.grad.float().norm().item() for n,p in m.native.named_parameters()
                if n in ('resid_lambdas','x0_lambdas','smear_lambda','backout_lambda','smear_gate.weight')
                or 'value_embeds.' in n or 've_gate.' in n]
    assert len(features)>5 and all(v>0 for v in features)


def test_last_short_layer_cached_decode_crosses_window_boundary():
    m = adapter(); original.exercise_features(m.native)
    ids = torch.randint(0,256,(1,521),device='cuda')
    cache = KVCache(1,2,600,64,8,'cuda',torch.bfloat16)
    with torch.no_grad():
        pre = m(ids[:,:520],kv_cache=cache)
        one = m(ids[:,520:],kv_cache=cache)
        full = m(ids)
    assert cache.get_pos() == 521
    torch.testing.assert_close(pre,full[:,:520],atol=.004,rtol=.03)
    torch.testing.assert_close(one,full[:,520:],atol=.004,rtol=.03)


def test_cce_compile_recompute_optimizer_rng_restore(tmp_path, monkeypatch):
    # Reuse the established native test and numerical thresholds with the new
    # 8-layer layout, including all native features and fixed-gradient restore.
    monkeypatch.setattr(original,'adapter',lambda variant,compile_blocks=False: adapter())
    original.test_cce_compile_checkpoint_and_resume('baseline',tmp_path)
