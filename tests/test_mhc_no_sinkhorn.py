"""Independent signed-mixing oracle and integration checks for the ablation."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for
from nanochat.engine import KVCache
from nanochat.nvidia_connections import NvidiaHyperConnection
from nanochat.nvidia_ops import nvidia_ops
from nanochat.packed_training import seed_all
from test_connection_training_resume import test_optimizer_rng_checkpoint_and_fixed_gradient_next_update as check_resume


def relative(a, b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()


def record(name, data):
    path = Path('results/no-sinkhorn-checks')
    path.mkdir(parents=True, exist_ok=True)
    (path/(name+'.json')).write_text(json.dumps(data, indent=2)+'\n')


def forbid_sinkhorn(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('The no-Sinkhorn path called Sinkhorn')
    monkeypatch.setattr(nvidia_ops(), 'mhc_fused_sinkhorn', forbidden)
    monkeypatch.setattr(nvidia_ops(), 'mhc_generate_mix_and_aggregate', forbidden)


@pytest.mark.parametrize('width', [128, 512])
@pytest.mark.parametrize('fused', [False, True])
def test_no_sinkhorn_pipeline_fp64(width, fused, monkeypatch):
    forbid_sinkhorn(monkeypatch)
    torch.manual_seed(352)
    m = NvidiaHyperConnection(width, 4, 0, sinkhorn=False, fused_grad_acc=fused).cuda()
    m.reset_parameters()
    x = torch.randn(2, 128, width, 4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    branch, residual, mixes = m.read(x)
    y = torch.tanh(branch.float()).to(x.dtype)
    out = m.write(y, residual, mixes)
    p = torch.randn_like(out).float()/out.numel()
    q = torch.randn_like(branch).float()/branch.numel()
    actual = torch.autograd.grad((out.float()*p).sum()+(branch.float()*q).sum(), (x, *m.parameters()))
    xr = x.detach().double().requires_grad_()
    phi, bias, scales = [v.detach().double().requires_grad_() for v in m.parameters()]
    flat = xr.flatten(-2)
    raw = F.linear(flat/(flat.square().mean(-1, keepdim=True)+torch.finfo(torch.float32).eps).sqrt(), phi)
    pre = (raw[..., :4]*scales[0]+bias[:4]).sigmoid()
    post = 2*(raw[..., 4:8]*scales[1]+bias[4:8]).sigmoid()
    comb = (raw[..., 8:]*scales[2]+bias[8:]).reshape(2, 128, 4, 4)
    br = (xr*pre.unsqueeze(-2)).sum(-1)
    expected = torch.einsum('btcn,btnm->btcm', xr, comb)+br.tanh().unsqueeze(-1)*post.unsqueeze(-2)
    reference = torch.autograd.grad((expected*p.double()).sum()+(br*q.double()).sum(), (xr, phi, bias, scales))
    errors = dict(output=relative(out, expected), branch=relative(branch, br),
                  residual_matrix=relative(mixes[1], comb),
                  gradients=[relative(a, b) for a, b in zip(actual, reference)],
                  minimum_residual_weight=mixes[1].min().item(),
                  row_sum_deviation=(mixes[1].sum(-1)-1).abs().max().item(),
                  column_sum_deviation=(mixes[1].sum(-2)-1).abs().max().item(),
                  sinkhorn_calls=0)
    record(f'pipeline-d{width}-fused{int(fused)}', errors)
    assert errors['output'] < .01 and errors['branch'] < .01, errors
    assert errors['residual_matrix'] < .001, errors
    assert max(errors['gradients']) < .03, errors
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in actual)
    assert errors['minimum_residual_weight'] < 0
    assert errors['row_sum_deviation'] > .001 and errors['column_sum_deviation'] > .001


def test_all_initial_parameters_match_mhc():
    models = []
    for variant in ('mhc', 'mhc_no_sinkhorn'):
        seed_all(42)
        model = ConnectionModel(ConnectionConfig(variant=variant)).prepare()
        match_vanilla_initialization(model, 42)
        models.append(model)
    a, b = [m.state_dict() for m in models]
    assert a.keys() == b.keys()
    assert all(torch.equal(a[name], b[name]) for name in a)
    left, right = [asdict(m.config) for m in models]
    assert {k for k in left if left[k] != right[k]} == {'variant'}
    for block in models[1].blocks:
        assert not block.attn_connection.sinkhorn and not block.mlp_connection.sinkhorn
    assert ConnectionConfig.from_checkpoint(right) == models[1].config
    # Distinct model identity prevents accidental resume across the ablation.
    assert ConnectionConfig.from_checkpoint(left) != models[1].config
    groups = []
    for m in models:
        opt, wd = optimizer_for(m)
        groups.append(([{k: v for k, v in g.items() if k != 'params'} for g in opt.param_groups], wd))
    assert groups[0] == groups[1]
    record('initialization', dict(all_parameters_bitwise_equal=True,
        optimizer_groups_equal=True, parameters=sum(p.numel() for p in models[1].parameters())))


def test_no_sinkhorn_restore_and_fixed_gradient_update(tmp_path):
    check_resume('mhc_no_sinkhorn', tmp_path)


def test_full_model_causal_cache_cce_checkpoint(monkeypatch):
    configure_attention('sdpa_cudnn')
    forbid_sinkhorn(monkeypatch)
    seed_all(42)
    config = ConnectionConfig(variant='mhc_no_sinkhorn', width=128, heads=2, depth=2,
                              vocab_size=128, sequence_length=2048)
    model = ConnectionModel(config).prepare()
    with torch.no_grad():
        for b in model.blocks:
            b.attn.c_proj.weight.normal_(std=.03)
            b.mlp.c_proj.weight.normal_(std=.03)
    ids = torch.randint(0, 128, (1, 2048), device='cuda')
    changed = ids.clone()
    changed[:, 1024:] = (changed[:, 1024:]+17) % 128
    with torch.no_grad():
        a, b = model(ids), model(changed)
        torch.testing.assert_close(a[:, :1024], b[:, :1024], atol=0, rtol=0)
        cache = KVCache(1, 2, 32, 64, 2, 'cuda', torch.bfloat16)
        pre = model(ids[:, :7], kv_cache=cache)
        one = model(ids[:, 7:8], kv_cache=cache)
        full = model(ids[:, :8])
        torch.testing.assert_close(pre, full[:, :7], atol=.004, rtol=.03)
        torch.testing.assert_close(one, full[:, 7:], atol=.004, rtol=.03)
    x, y = ids[:, :128].contiguous(), ids[:, 1:129].contiguous()
    loss = model(x, y)
    dense = model(x, y, dense=True)
    torch.testing.assert_close(loss, dense, atol=2e-4, rtol=2e-4)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    grads = {k: p.grad.clone() for k, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    model.config.activation_checkpoint = True
    again = model(x, y)
    again.backward()
    errors = {k: relative(p.grad, grads[k]) for k, p in model.named_parameters()}
    record('model', dict(causal_tokens=2048, sinkhorn_calls=0, cce_loss=loss.item(),
        dense_loss=dense.item(), checkpoint_gradient_relative_errors=errors))
    torch.testing.assert_close(loss, again, atol=2e-5, rtol=2e-5)
    assert max(errors.values()) < .02, errors
