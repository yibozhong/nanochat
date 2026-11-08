"""NVIDIA-layout implementation versus the prior accepted scale implementation."""
import pytest
import torch
from nanochat.scaled_mixing import scaled_mix
from nanochat.scaled_connections import ScaledHyperConnection
from scripts.connection_scaled_hc_nvidia_profile import legacy_function
from test_scaled_hc import inputs,relative,model,record


@pytest.fixture(autouse=True)
def fresh_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize('n,c',[(129,128),(2048,512),(32768,512)])
def test_legacy_all_vjps(n,c):
    torch.manual_seed(29)
    args=inputs(n,c);go=torch.randn_like(args[0])
    a=scaled_mix(*args);ga=torch.autograd.grad(a,args,go)
    b=legacy_function()(*args);gb=torch.autograd.grad(b,args,go)
    errors=dict(output=relative(a,b),gradients=[relative(x,y) for x,y in zip(ga,gb)])
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    record(f'nvidia-legacy-{n}-{c}',errors)


def test_fullmodel_legacy_gradients(monkeypatch):
    m=model();x=torch.randint(0,128,(2,128),device='cuda');y=(x+1)%128
    loss=m(x,y);loss.backward();grads={n:p.grad.clone() for n,p in m.named_parameters()}
    legacy=legacy_function()
    def write(self,output,residual,mixes):
        post,h,_=mixes
        return legacy(residual,output.contiguous(),post.contiguous(),h.contiguous(),self.input_scale)
    monkeypatch.setattr(ScaledHyperConnection,'write',write)
    m.zero_grad(set_to_none=True);other=m(x,y);other.backward()
    errors={n:relative(grads[n],p.grad) for n,p in m.named_parameters()}
    torch.testing.assert_close(loss,other,atol=2e-4,rtol=2e-4)
    assert max(errors.values())<.02,errors
    record('nvidia-fullmodel-legacy',dict(loss=loss.item(),legacy=other.item(),gradients=errors))
