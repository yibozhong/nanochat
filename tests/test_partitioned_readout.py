"""Independent full-softmax oracles and persistent-stream integration checks."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from nanochat.partitioned_readout import vocab_partition, combine_partition_losses, partitioned_cross_entropy, partitioned_logits
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for
from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.packed_training import seed_all
from nanochat.engine import KVCache
import test_connection_training_resume as resume_checks


def record(name, data):
    folder = Path('results/partition-checks'); folder.mkdir(parents=True, exist_ok=True)
    (folder/(name+'.json')).write_text(json.dumps(data, indent=2)+'\n')


def relative(a, b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()


def test_fp64_global_objective_gradient_and_rank():
    torch.manual_seed(52)
    n, k, d, v = 11, 4, 5, 48
    ids, inverse = vocab_partition(v, k, 12)
    h = torch.randn(n, k, d, dtype=torch.float64, requires_grad=True)
    w = torch.randn(v, d, dtype=torch.float64, requires_grad=True)
    targets = ids[0, torch.arange(n)]  # All targets in stream zero; other streams still get gradients.
    z = [15*torch.tanh(F.linear(h[:, i], w[part])/15) for i, part in enumerate(ids)]
    grouped = inverse[targets]
    local = [F.cross_entropy(a, grouped % (v//k), reduction='none') for a in z]
    actual = combine_partition_losses(local, [a.logsumexp(-1) for a in z], grouped//(v//k))
    expected = F.cross_entropy(torch.cat(z, -1)[:, inverse], targets)
    a = torch.autograd.grad(actual, (h, w), retain_graph=True)
    b = torch.autograd.grad(expected, (h, w))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for x, y in zip(a, b): torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-12)
    norms = a[0].norm(dim=(0, 2))
    assert (norms > 0).all()
    matrix = torch.block_diag(*[w[part].detach() for part in ids])
    assert torch.linalg.matrix_rank(matrix).item() == k*d
    assert torch.linalg.matrix_rank(w.detach()).item() == d
    record('fp64', dict(loss_error=abs((actual-expected).item()),
        gradient_errors=[relative(x, y) for x, y in zip(a,b)], stream_gradient_norms=norms.tolist(),
        example_partition_rank=k*d, example_collapsed_rank=d))


@pytest.mark.parametrize('scale', [.1, 1.])
@pytest.mark.parametrize('one_partition', [False, True])
def test_cce_global_loss_and_both_gradients(scale, one_partition):
    torch.manual_seed(123)
    n, k, d, v = 256, 4, 128, 1024
    ids, inverse = [a.cuda() for a in vocab_partition(v, k, 19)]
    h = (torch.randn(n, k, d, device='cuda')*scale).bfloat16().requires_grad_()
    w = (torch.randn(v, d, device='cuda')*.2).bfloat16().requires_grad_()
    targets = torch.randint(0, v//k if one_partition else v, (n,), device='cuda')
    if one_partition: targets = ids[0, targets]
    actual = partitioned_cross_entropy(h, w, targets, ids, inverse)
    ga = torch.autograd.grad(actual, (h,w))
    hr = h.detach().float().requires_grad_(); wr = w.detach().float().requires_grad_()
    expected = F.cross_entropy(partitioned_logits(hr,wr,ids,inverse), targets)
    gr = torch.autograd.grad(expected, (hr,wr))
    errors = [relative(a,b) for a,b in zip(ga,gr)]
    stream_norms = ga[0].float().norm(dim=(0,2))
    value = dict(cce_loss=actual.item(), dense_loss=expected.item(), gradient_relative_errors=errors,
                 stream_gradient_norms=stream_norms.tolist())
    record(f'cce-{scale}-{one_partition}', value)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    assert max(errors) < .012, value  # BF16 GEMM/gradient downcasts, FP32 accumulation.
    assert (stream_norms > 0).all()


def build(readout, **kwargs):
    seed_all(42)
    model = ConnectionModel(ConnectionConfig(variant='mhc', readout=readout, **kwargs)).prepare()
    match_vanilla_initialization(model, 42)
    return model


def test_matched_initialization_partition_and_optimizer():
    left, right = [build(r) for r in ('collapse', 'vocab_partition')]
    a, b = left.state_dict(), right.state_dict()
    assert set(b)-set(a) == {'partition_ids', 'partition_inverse'}
    assert {name for name in a if name not in b} == {'stream_head.phi', 'stream_head.bias', 'stream_head.scale'}
    assert all(torch.equal(a[name], b[name]) for name in a.keys() & b.keys())
    assert right.hidden(torch.zeros(1,32,dtype=torch.long,device='cuda')).shape == (1,32,4,512)
    ids = right.partition_ids.flatten()
    assert torch.equal(ids[right.partition_inverse], torch.arange(32768,device='cuda'))
    assert torch.equal(vocab_partition(32768,4,20261031)[0], right.partition_ids.cpu())
    groups = []
    for m in (left, right):
        opt, wd = optimizer_for(m)
        groups.append(([{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups],wd))
    assert groups[0] == groups[1]
    counts = [sum(p.numel() for p in m.parameters()) for m in (left,right)]
    assert counts == [59515317,59507120], counts
    assert ConnectionConfig.from_checkpoint(asdict(right.config)) == right.config
    record('initialization', dict(shared_parameters_bitwise_equal=True, optimizer_hyperparameters_equal=True,
                                 parameters=counts, head_parameters=32768*512, partition_seed=20261031))


def test_causal_cache_compile_and_checkpoint():
    configure_attention('sdpa_cudnn')
    model = build('vocab_partition', depth=2,width=128,heads=2,vocab_size=256)
    with torch.no_grad():
        for b in model.blocks:
            b.attn.c_proj.weight.normal_(std=.03); b.mlp.c_proj.weight.normal_(std=.03)
    ids = torch.randint(0,256,(1,2048),device='cuda')
    changed = ids.clone(); changed[:,1024:] = (changed[:,1024:]+31)%256
    with torch.no_grad():
        a,b = model(ids),model(changed)
        torch.testing.assert_close(a[:,:1024],b[:,:1024],atol=0,rtol=0)
        cache = KVCache(1,2,32,64,2,'cuda',torch.bfloat16)
        pre = model(ids[:,:7],kv_cache=cache); one = model(ids[:,7:8],kv_cache=cache)
        full = model(ids[:,:8])
        torch.testing.assert_close(pre,full[:,:7],atol=.004,rtol=.03)
        torch.testing.assert_close(one,full[:,7:],atol=.004,rtol=.03)
    model.prepare(compile_blocks=True)
    x,y = ids[:,:128].contiguous(),ids[:,1:129].contiguous()
    loss = model(x,y); dense = model(x,y,dense=True)
    torch.testing.assert_close(loss,dense,atol=2e-4,rtol=2e-4)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    grads = {name:p.grad.clone() for name,p in model.named_parameters()}
    model.zero_grad(set_to_none=True); model.config.activation_checkpoint=True
    again = model(x,y); again.backward()
    errors = {name:relative(p.grad,grads[name]) for name,p in model.named_parameters()}
    record('integration',dict(causal_tokens=2048,cce_loss=loss.item(),dense_loss=dense.item(),
                             checkpoint_gradient_relative_errors=errors))
    assert max(errors.values()) < .02, errors


def test_partition_optimizer_rng_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(resume_checks,'build',lambda variant:build('vocab_partition',
        depth=2,width=128,heads=1,vocab_size=256,sequence_length=128))
    resume_checks.test_optimizer_rng_checkpoint_and_fixed_gradient_next_update('mhc',tmp_path)
