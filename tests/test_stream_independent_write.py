"""The corrected architecture must retain four updates until the MLP read."""
from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from nanochat.connection_model import ConnectionConfig, optimizer_for, schedule
from nanochat.nvidia_connections import NvidiaHyperConnection, NvidiaStreamConnection
from nanochat.packed_training import seed_all
from nanochat.stream_attention import StreamAttention
from test_mhc_nvidia_full import relative, reference_mix, record
from test_stream_attention import build, attention_policy
from test_connection_training_resume import equal_tree


def test_independent_output_and_gradient_stream_isolation():
    seed_all(108)
    module = StreamAttention(128, 4, 4, 0, dense_parameters=True).cuda().double()
    x = torch.randn(2, 9, 128, 4, device='cuda', dtype=torch.float64, requires_grad=True)
    cos = torch.ones(1, 9, 1, 16, device='cuda', dtype=torch.float64)
    with sdpa_kernel(SDPBackend.MATH):
        y = module(x, None, (cos, cos*0), return_streams=True)
        changed = x.detach().clone(); changed[..., 1:] += 10
        other = module(changed, None, (cos, cos*0), return_streams=True)
    assert y.shape == x.shape
    torch.testing.assert_close(y[..., 0], other[..., 0], atol=0, rtol=0)
    y[..., 0].square().sum().backward()
    assert x.grad[..., 0].norm() > 0 and torch.count_nonzero(x.grad[..., 1:]) == 0
    for layer in (module.c_q, module.c_k, module.c_v):
        assert layer.weight.grad[:32].norm() > 0
        assert torch.count_nonzero(layer.weight.grad[32:]) == 0
    assert module.c_proj.weight.grad[:, :32].norm() > 0
    assert torch.count_nonzero(module.c_proj.weight.grad[:, 32:]) == 0


@pytest.mark.parametrize('sinkhorn', [True, False])
def test_independent_nvidia_write_fp64_forward_backward(sinkhorn):
    seed_all(45)
    old = NvidiaHyperConnection(128, 4, 0, sinkhorn=sinkhorn).cuda(); old.reset_parameters()
    layer = NvidiaStreamConnection(old)
    x = torch.randn(2, 64, 128, 4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    # Independent, unequal branch outputs, not an already collapsed vector.
    y = torch.randn_like(x, requires_grad=True)
    skip, mix = layer.routing(x); actual = layer.write(y, skip, mix)
    xr = x.detach().double().requires_grad_(); yr = y.detach().double().requires_grad_()
    params = [p.detach().double().requires_grad_() for p in layer.parameters()]
    phi, bias, scales = params
    phi24, bias24, scales3 = F.pad(phi, (0, 0, 4, 0)), F.pad(bias, (4, 0)), F.pad(scales, (1, 0))
    _, post, comb = reference_mix(xr, phi24, scales3, bias24)
    if not sinkhorn:
        flat = xr.flatten(-2)
        raw = F.linear(flat/(flat.square().mean(-1, keepdim=True)+torch.finfo(torch.float32).eps).sqrt(), phi24)
        comb = (raw[..., 8:]*scales3[2]+bias24[8:]).reshape(2, 64, 4, 4)
    expected = torch.einsum('btcn,btnm->btcm', xr, comb) + yr*post.unsqueeze(-2)
    probe = torch.randn_like(actual).float()/actual.numel()
    ga = torch.autograd.grad((actual.float()*probe).sum(), (x, y, *layer.parameters()))
    gr = torch.autograd.grad((expected*probe).sum(), (xr, yr, *params))
    errors = dict(output=relative(actual, expected), gradients=[relative(a, b) for a, b in zip(ga, gr)])
    record(f'independent-write-sinkhorn{sinkhorn}', errors)
    assert errors['output'] < .01 and max(errors['gradients']) < .03, errors


def test_branch_write_does_not_broadcast_to_other_streams():
    old = NvidiaHyperConnection(128, 4, 0).cuda(); old.reset_parameters()
    layer = NvidiaStreamConnection(old)
    x = torch.zeros(2, 64, 128, 4, device='cuda', dtype=torch.bfloat16)
    y = torch.zeros_like(x, requires_grad=True)
    with torch.no_grad(): y[..., 2].fill_(1)
    post = torch.ones(2, 64, 4, device='cuda')
    identity = torch.eye(4, device='cuda').expand(2, 64, 4, 4).contiguous()
    out = layer.write(y, x, (post, identity, None))
    assert torch.equal(out, y)
    out[..., 2].sum().backward()
    assert torch.count_nonzero(y.grad[..., [0, 1, 3]]) == 0
    assert torch.equal(y.grad[..., 2], torch.ones_like(y.grad[..., 2]))


def test_mlp_reads_four_independently_updated_streams():
    model = build(mode='per_stream_independent', length=64, depth=1)
    block = model.blocks[0]
    x = torch.randn(2, 64, 512, 4, device='cuda', dtype=torch.bfloat16)
    outputs = torch.randn_like(x)
    skip, mixes = block.attn_connection.routing(x)
    expected = block.attn_connection.write(outputs, skip, mixes)
    captured = []
    read = block.mlp_connection.read
    def capture(value):
        captured.append(value)
        return read(value)
    with patch.object(block, 'attention_branch', return_value=outputs), \
         patch.object(block.mlp_connection, 'read', side_effect=capture):
        block(x, model.cos, model.sin)
    assert len(captured) == 1
    assert captured[0].shape == (2, 64, 512, 4)
    torch.testing.assert_close(captured[0], expected, atol=0, rtol=0)
    post, _, _ = mixes
    original_shared_write = (outputs.float().sum(-1).unsqueeze(-1)*post.unsqueeze(-2))
    diagonal_write = outputs.float()*post.unsqueeze(-2)
    assert not torch.allclose(original_shared_write, diagonal_write)


def test_no_unused_attention_pre_parameters_and_explicit_checkpoint_mode():
    old = build(mode='per_stream', length=32)
    new = build(mode='per_stream_independent', length=32)
    shared = build(mode='shared', length=32)
    ids = torch.randint(0, 256, (2, 32), device='cuda')
    with torch.no_grad():
        torch.testing.assert_close(new(ids), shared(ids), atol=0, rtol=0)
    for name, p in new.named_parameters():
        if '.attn_connection.' not in name and '.attn.' not in name:
            assert torch.equal(p, old.get_parameter(name)), name
    for a, b in zip(old.blocks, new.blocks):
        for name in ('c_q', 'c_k', 'c_v', 'c_proj'):
            axis = 1 if name == 'c_proj' else 0
            assert torch.equal(torch.cat([p.weight for p in getattr(a.attn, name)], dim=axis),
                               getattr(b.attn, name).weight)
        assert torch.equal(a.attn_connection.phi[4:], b.attn_connection.phi)
        assert torch.equal(a.attn_connection.bias[4:], b.attn_connection.bias)
        assert torch.equal(a.attn_connection.scales[1:], b.attn_connection.scales)
    removed = len(new.blocks)*(4*(4*512)+4+1)
    assert sum(p.numel() for p in old.parameters())-sum(p.numel() for p in new.parameters()) == removed
    config = asdict(new.config)
    assert ConnectionConfig.from_checkpoint(config).attention_mode == 'per_stream_independent'
    assert ConnectionConfig.from_checkpoint(asdict(old.config)).attention_mode == 'per_stream'
    opt, _ = optimizer_for(new)
    from torch._dynamo import config as compiler_config
    signatures = {(tuple(p.shape), p.dtype, tuple(p.stride()), p.device.type)
                  for g in opt.param_groups if g['kind']=='adamw' for p in g['params']}
    assert len(signatures) > 8 and compiler_config.recompile_limit >= 2*len(signatures)
    grouped = [id(p) for g in opt.param_groups for p in g['params']]
    assert len(grouped) == len(set(grouped)) == len(list(new.parameters()))
    with pytest.raises(RuntimeError): new.load_state_dict(old.state_dict())


def test_dense_muon_same_gradients_match_shared_update_and_moments():
    """Control optimizer geometry, including warmup, momentum, decay and resume."""
    shared = build(mode='shared', length=32, depth=2)
    independent = build(mode='per_stream_independent', length=32, depth=2)
    left, wd = optimizer_for(shared); right, other_wd = optimizer_for(independent)
    assert wd == other_wd
    groups_l = [g for g in left.param_groups if g['kind'] == 'muon']
    groups_r = [g for g in right.param_groups if g['kind'] == 'muon']
    assert len(groups_l) == len(groups_r)
    # Different stream slices receive different gradients, but are updated as
    # the same square matrix used by the original mHC optimizer.
    seed_all(871)
    for step in (0, 1, 399, 400, 120000):
        for opt in (left, right): schedule(opt, step, 305176, wd)
        for model in (shared, independent):
            for p in model.parameters(): p.grad = torch.zeros_like(p)
        for gl, gr in zip(groups_l, groups_r):
            assert {k:v for k,v in gl.items() if k!='params'} == {k:v for k,v in gr.items() if k!='params'}
            assert len(gl['params']) == len(gr['params'])
            for a, b in zip(gl['params'], gr['params']):
                assert a.shape == b.shape and torch.equal(a, b)
                grad = torch.randn_like(a)
                a.grad = grad.clone(); b.grad = grad.clone()
        left.step(); right.step()
        for gl, gr in zip(groups_l, groups_r):
            for a, b in zip(gl['params'], gr['params']): assert torch.equal(a, b)
            equal_tree(left.state[gl['params'][0]], right.state[gr['params'][0]])
    assert all(p.shape == (512, 512) for b in independent.blocks for p in b.attn.parameters())
    record('independent-dense-muon-control', dict(passed=True, exact_parameter_and_state_match=True,
        checked_schedule_steps=[0, 1, 399, 400, 120000], attention_shape=[512, 512]))
