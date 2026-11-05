"""Verify the selected A100 attention kernel and its gradients, without a baseline run."""
import torch
from torch.nn import functional as F

from nanochat.connection_experiment import configure_attention
from nanochat.nvidia_connections import check_nvidia_device
from nanochat.nvidia_ops import nvidia_ops
from test_stream_mlp import relative, record


def test_a100_fa2_dispatch_and_nontma_projection():
    assert torch.cuda.get_device_capability() == (8, 0)
    assert 'A100' in torch.cuda.get_device_name()
    check_nvidia_device('cuda')
    configure_attention('fa2')
    from nanochat import flash_attention
    assert not flash_attention.USE_FA3 and not flash_attention.HAS_FA3
    assert torch.backends.cuda.flash_sdp_enabled()
    assert not torch.backends.cuda.math_sdp_enabled()
    assert not torch.backends.cuda.mem_efficient_sdp_enabled()
    assert not torch.backends.cuda.cudnn_sdp_enabled()
    torch.manual_seed(72)
    args=[torch.randn(2,4,29,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    assert not nvidia_ops()._support_tma(args[0])
    go=torch.randn_like(args[0])
    # Warm up outside the trace; no compiler/backend fallback is permitted.
    torch.autograd.grad(F.scaled_dot_product_attention(*args,is_causal=True),args,go)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as trace:
        actual=F.scaled_dot_product_attention(*args,is_causal=True)
        grads=torch.autograd.grad(actual,args,go)
    keys=[e.key for e in trace.key_averages()]
    assert 'aten::_scaled_dot_product_flash_attention' in keys,keys
    assert 'aten::_scaled_dot_product_flash_attention_backward' in keys,keys
    oracle=[a.detach().double().requires_grad_() for a in args]
    q,k,v=oracle
    logits=q@k.transpose(-1,-2)/(q.shape[-1]**.5)
    mask=torch.ones(29,29,device='cuda',dtype=torch.bool).triu(1)
    expected=logits.masked_fill(mask,float('-inf')).softmax(-1)@v
    eg=torch.autograd.grad(expected,oracle,go.double())
    errors=dict(output=relative(actual,expected),gradients=[relative(a,b) for a,b in zip(grads,eg)])
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    record('a100-fa2',dict(backend='torch2.10 SDPA FLASH_ATTENTION (FA2)',projection_tma=False,
                          operators=keys,errors=errors))
