"""Verify hardware-selected attention and gradients, without a baseline run."""
import torch

from nanochat.connection_experiment import configure_attention, resolve_attention_backend
from nanochat.nvidia_connections import check_nvidia_device
from nanochat.nvidia_ops import nvidia_ops
from test_stream_mlp import relative, record


def test_hardware_dispatch_attention_gradient_and_projection():
    capability=torch.cuda.get_device_capability()
    expected_backend='fa3' if capability[0]==9 else 'fa2'
    assert resolve_attention_backend('hardware_auto')==expected_backend
    check_nvidia_device('cuda')
    assert configure_attention('hardware_auto')==expected_backend
    from nanochat import flash_attention
    if expected_backend=='fa3':
        assert flash_attention.USE_FA3 and flash_attention.HAS_FA3
    else:
        assert not flash_attention.USE_FA3 and not flash_attention.HAS_FA3
        assert torch.backends.cuda.flash_sdp_enabled()
        assert not torch.backends.cuda.math_sdp_enabled()
        assert not torch.backends.cuda.mem_efficient_sdp_enabled()
        assert not torch.backends.cuda.cudnn_sdp_enabled()
    torch.manual_seed(72)
    args=[torch.randn(2,4,29,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    use_tma=nvidia_ops()._support_tma(args[0])
    assert use_tma==(capability[0]==9)
    go=torch.randn_like(args[0])
    def attention(q,k,v):
        # Exercise the model's actual public interface on both hardware paths.
        return flash_attention.flash_attn_func(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),causal=True).transpose(1,2)
    # Warm up outside the trace; no compiler/backend fallback is permitted.
    torch.autograd.grad(attention(*args),args,go)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as trace:
        actual=attention(*args)
        grads=torch.autograd.grad(actual,args,go)
    keys=[e.key for e in trace.key_averages()]
    if expected_backend=='fa2':
        assert 'aten::_scaled_dot_product_flash_attention' in keys,keys
        assert 'aten::_scaled_dot_product_flash_attention_backward' in keys,keys
    else:
        assert any('flash' in k.lower() and ('fwd' in k.lower() or 'forward' in k.lower()) for k in keys),keys
        assert any('flash' in k.lower() and ('bwd' in k.lower() or 'backward' in k.lower()) for k in keys),keys
    oracle=[a.detach().double().requires_grad_() for a in args]
    q,k,v=oracle
    logits=q@k.transpose(-1,-2)/(q.shape[-1]**.5)
    mask=torch.ones(29,29,device='cuda',dtype=torch.bool).triu(1)
    expected=logits.masked_fill(mask,float('-inf')).softmax(-1)@v
    eg=torch.autograd.grad(expected,oracle,go.double())
    errors=dict(output=relative(actual,expected),gradients=[relative(a,b) for a,b in zip(grads,eg)])
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    record('hardware-backend',dict(backend=expected_backend,projection_tma=use_tma,
                          operators=keys,errors=errors))
