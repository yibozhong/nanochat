"""Cross-pass gradient oracle and regression checks, run on a GPU worker."""
from dataclasses import asdict,replace
from unittest.mock import patch

import pytest
import torch

from nanochat.connection_cost import model_cost
from nanochat.connection_model import ConnectionConfig,build_connection_model
from nanochat.layer_feedback import layer_read,layer_read_reference
from nanochat.packed_training import seed_all
from test_projected_feedback import gpu,relative,make,inputs,MODES


@pytest.mark.parametrize('shape',[(2,17,3,128),(2,1,8,512),(2,9,5,80),(2,9,1,128)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.float16])
@pytest.mark.parametrize('shift',[False,True])
@pytest.mark.parametrize('normalize',[False,True])
def test_bank_gradient_oracle(shape,dtype,shift,normalize):
    seed_all(34);b,t,l,d=shape
    q=torch.randn(b,t,d,device='cuda',dtype=dtype,requires_grad=True)
    bank=torch.randn(b,t,l,d,device='cuda',dtype=dtype,requires_grad=True)
    go=torch.randn_like(q)
    out=layer_read(q,bank,shift=shift,normalize=normalize)
    grads=torch.autograd.grad(out,(q,bank),go)
    rq=q.detach().double().requires_grad_();rb=bank.detach().double().requires_grad_()
    ref=layer_read_reference(rq,rb,shift=shift,normalize=normalize)
    expected=torch.autograd.grad(ref,(rq,rb),go.double())
    tol=.009 if dtype==torch.bfloat16 else .002 if dtype==torch.float16 else 3e-5
    for actual,want in zip((out,*grads),(ref,*expected)):
        assert torch.isfinite(actual).all()
        assert relative(actual,want.to(dtype))<tol
    if shift:
        assert torch.count_nonzero(grads[1][:,-1])==0


def test_production_bank_gradient():
    seed_all(34)
    q=torch.randn(16,2048,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    bank=torch.randn(16,2048,8,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    go=torch.randn_like(q)
    a=layer_read(q,bank,normalize=False);ga=torch.autograd.grad(a,(q,bank),go)
    b=layer_read_reference(q,bank,normalize=False);gb=torch.autograd.grad(b,(q,bank),go)
    for actual,want in zip((a,*ga),(b,*gb)):assert relative(actual,want)<.009


def connected(mode,backend='triton',compiled=False,checkpoint=False):
    template=make(mode)
    c=replace(template.config,feedback_detach=False,feedback_attention_backend=backend,
        activation_checkpoint=checkpoint)
    m=build_connection_model(c).prepare(compile_blocks=compiled)
    m.load_state_dict(template.state_dict())
    return m


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('compiled,checkpoint,passes',[(False,False,2),(True,False,3),(True,True,2)])
def test_complete_model_gradients(mode,compiled,checkpoint,passes):
    torch._dynamo.config.recompile_limit=max(32,torch._dynamo.config.recompile_limit)
    m=connected(mode,compiled=compiled,checkpoint=checkpoint)
    # Compare the read kernels with the same compilation/rounding regime.
    # Compiled vs eager BF16 norm/backward can differ across repeated passes.
    ref=connected(mode,backend='torch',compiled=compiled,checkpoint=checkpoint)
    ref.load_state_dict(m.state_dict())
    x,y=inputs();a=m(x,y,passes=passes);b=ref(x,y,passes=passes)
    a.backward();b.backward()
    torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-4)
    for (n,p),(_,r) in zip(m.named_parameters(),ref.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(),n
        assert relative(p.grad,r.grad)<.04,(n,relative(p.grad,r.grad))


@pytest.mark.parametrize('mode',MODES)
def test_bank_receives_gradient_head_once_forward_unchanged(mode):
    m=connected(mode);detached=make(mode);detached.load_state_dict(m.state_dict())
    x,y=inputs();banks=[];fn=m.backbone
    def observe(*args,**kwargs):
        out=fn(*args,**kwargs)
        if isinstance(out,tuple):
            assert torch.is_grad_enabled() and out[1].requires_grad
            out[1].retain_grad();banks.append(out[1])
        return out
    with patch.object(m,'backbone',side_effect=observe),patch.object(m,'readout',wraps=m.readout) as head:
        a=m(x,y,passes=3);a.backward();assert head.call_count==1
    assert len(banks)==2
    for bank in banks:
        assert bank.grad is not None and bank.grad.norm()>0
        assert torch.count_nonzero(bank.grad[:,-1])==0
    with torch.no_grad():
        assert torch.equal(m(x,passes=3),detached(x,passes=3))
        other=x.clone();other[:,7:]=(other[:,7:]+31)%256
        assert torch.equal(m(x,passes=3)[:,:7],m(other,passes=3)[:,:7])
    m.zero_grad(set_to_none=True)
    m(x,y,dense=True,passes=1).backward();detached(x,y,dense=True,passes=1).backward()
    for p,r in zip(m.parameters(),detached.parameters()):
        if p.grad is None:assert r.grad is None
        else:assert torch.equal(p.grad,r.grad)


@pytest.mark.parametrize('mode',MODES)
def test_config_roundtrip_and_cost(mode):
    c=replace(make(mode).config,feedback_detach=False)
    assert ConnectionConfig.from_checkpoint(asdict(c))==c
    legacy=asdict(c);legacy.pop('feedback_detach')
    assert ConnectionConfig.from_checkpoint(legacy).feedback_detach
    base=model_cost(replace(c,feedback_detach=True),feedback_passes=1)
    p1=model_cost(c,feedback_passes=1);p2=model_cost(c,feedback_passes=2)
    assert p1['estimated_training_flops']==base['estimated_training_flops']
    assert p2['estimated_training_flops']>model_cost(replace(c,feedback_detach=True),feedback_passes=2)['estimated_training_flops']
    assert model_cost(c)['estimated_training_flops']==pytest.approx(.15*p1['estimated_training_flops']+.85*p2['estimated_training_flops'])
