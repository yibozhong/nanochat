"""Numerical and causal checks for replacing only the connection projection."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from nanochat.connection_model import ConnectionConfig, ConnectionModel
from nanochat.manifold_connections import projection, official_ops
from nanochat.engine import KVCache


def relative(a, b):
    return ((a.double()-b.double()).norm() / b.double().norm().clamp_min(1e-15)).item()


@pytest.mark.parametrize('backend', ['torch', 'torch_compile'])
@pytest.mark.parametrize('width', [128, 512])
def test_projection_against_fp64_and_official(backend, width):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(123)
    x = torch.randn(2, 256, 4, width, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(24, 4*width, device='cuda') / (4*width)**.5).requires_grad_()
    probe = torch.randn(2, 256, 24, device='cuda') / (2*256*24)
    actual = projection(x, w, backend)
    grads = torch.autograd.grad((actual*probe).sum(), (x, w))
    # Independent order in FP64: normalize the input before the linear layer.
    xr = x.detach().double().requires_grad_()
    wr = w.detach().double().requires_grad_()
    flat = xr.flatten(-2)
    expected = F.linear(flat / (flat.square().mean(-1, keepdim=True)+1e-6).sqrt(), wr)
    reference_grads = torch.autograd.grad((expected*probe.double()).sum(), (xr, wr))
    xo = x.detach().clone().requires_grad_()
    wo = w.detach().clone().requires_grad_()
    official = projection(xo, wo)
    official_grads = torch.autograd.grad((official*probe).sum(), (xo, wo))
    errors = dict(fp64_output=relative(actual, expected),
                  fp64_input_grad=relative(grads[0], reference_grads[0].to(torch.bfloat16)),
                  fp64_weight_grad=relative(grads[1], reference_grads[1]),
                  official_output=relative(actual, official),
                  official_gradients=[relative(a,b) for a,b in zip(grads,official_grads)])
    assert errors['fp64_output'] < (.003 if backend == 'nvidia' else 2e-5), errors
    assert errors['fp64_weight_grad'] < (.003 if backend == 'nvidia' else 3e-5), errors
    assert errors['fp64_input_grad'] < .005, errors
    assert errors['official_output'] < .003 and max(errors['official_gradients']) < .02, errors
    out=Path('results/torch-projection-checks');out.mkdir(parents=True,exist_ok=True)
    (out/f'{backend}-d{width}.json').write_text(json.dumps(errors,indent=2)+'\n')


@pytest.mark.parametrize('backend', ['torch', 'torch_compile'])
def test_torch_projection_model_causality_cache_and_all_gradients(backend):
    torch.backends.cuda.matmul.allow_tf32 = False
    opts=dict(variant='mhc', width=128, heads=2, depth=2, vocab_size=128, sequence_length=2048)
    torch.manual_seed(33)
    official=ConnectionModel(ConnectionConfig(**opts)).prepare()
    torch.manual_seed(33)
    model=ConnectionModel(ConnectionConfig(**opts, projection_backend=backend)).prepare()
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,official.state_dict()[key],atol=0,rtol=0)
    with torch.no_grad():
        for block in model.blocks:
            block.attn.c_proj.weight.normal_(std=.03)
            block.mlp.c_proj.weight.normal_(std=.03)
    official.load_state_dict(model.state_dict())
    ids=torch.randint(0,128,(1,2048),device='cuda')
    changed=ids.clone();changed[:,1024:]=(changed[:,1024:]+17)%128
    # no_grad must not silently fall back to the old fused projection.
    with patch.object(official_ops(),'mhc_pre_big_fuse',side_effect=AssertionError('Old projection used')):
        with torch.no_grad():
            a=model(ids);b=model(changed)
            torch.testing.assert_close(a[:,:1024],b[:,:1024],atol=0,rtol=0)
            cache=KVCache(1,2,32,64,2,'cuda',torch.bfloat16)
            pre=model(ids[:,:7],kv_cache=cache);next_token=model(ids[:,7:8],kv_cache=cache)
            full=model(ids[:,:8])
            torch.testing.assert_close(pre,full[:,:7],atol=.004,rtol=.03)
            torch.testing.assert_close(next_token,full[:,7:],atol=.004,rtol=.03)
        x,y=ids[:,:128].contiguous(),ids[:,1:129].contiguous()
        loss=model(x,y);dense=model(x,y,dense=True)
        torch.testing.assert_close(loss,dense,atol=2e-4,rtol=2e-4)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert sum(p.grad.float().square().sum().item() for n,p in model.named_parameters()
                   if '_connection.' in n)>0
    # Official vs PyTorch is tolerance-based: the former explicitly rounds TF32.
    reference=official(x,y)
    torch.testing.assert_close(loss,reference,atol=2e-3,rtol=2e-3)


def test_invalid_projection_selection_is_rejected():
    with pytest.raises(ValueError):ConnectionConfig(projection_backend='torch')
    with pytest.raises(ValueError):ConnectionConfig(variant='mhc',projection_backend='unknown')
