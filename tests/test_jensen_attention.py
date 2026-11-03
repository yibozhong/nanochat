"""Signed SiLU Jensen-gap formula and complete model integration checks."""
import pytest
import torch
from torch.nn import functional as F

from nanochat.moment_attention import value_moments
import test_moment_attention as shared

setup = shared.setup


def oracle(q,k,v,offset=0):
    outputs=[]
    for i in range(q.shape[1]):
        values=v[:,:offset+i+1]
        logits=torch.einsum('bhd,bshd->bhs',q[:,i],k[:,:offset+i+1])/q.shape[-1]**.5
        a=logits.softmax(-1).transpose(1,2).unsqueeze(-1)
        # Independent sigmoid expression, no call to production moment_tile.
        phi_values=values*torch.sigmoid(values)
        mean=(a*values).sum(1)
        outputs.append((a*phi_values).sum(1)-mean*torch.sigmoid(mean))
    return torch.stack(outputs,1)


@pytest.mark.parametrize('compiled',[False,True])
def test_formula_and_all_qkv_gradients(compiled):
    dtype=torch.float32 if compiled else torch.float64
    xs=[torch.randn(2,16,2,8,device='cuda',dtype=dtype,requires_grad=True) for _ in range(3)]
    fn=torch.compile(value_moments,fullgraph=True,dynamic=False) if compiled else value_moments
    actual=fn(*xs,centered=False,query_chunk=4,nonlinearity='silu')
    refs=[x.detach().double().requires_grad_() for x in xs]
    expected=oracle(*refs)
    tol=3e-6 if compiled else 3e-12
    torch.testing.assert_close(actual.double(),expected,atol=tol,rtol=tol)
    up=torch.randn_like(actual)
    ga=torch.autograd.grad((actual*up).sum(),xs)
    ge=torch.autograd.grad((expected*up.double()).sum(),refs)
    for a,e in zip(ga,ge):
        assert (a.double()-e).norm()/e.norm().clamp_min(1e-12)<tol


def test_signed_gap_and_zero_gap_cases():
    q=k=torch.zeros(1,2,1,2,device='cuda',dtype=torch.float64)
    # SiLU curvature is positive around zero, negative in this positive tail.
    v=torch.tensor([[-1.,3.],[1.,5.]],device='cuda',dtype=torch.float64).view(1,2,1,2)
    actual=value_moments(q,k,v,centered=False,nonlinearity='silu')
    assert actual[0,1,0,0]>0 and actual[0,1,0,1]<0
    torch.testing.assert_close(actual,oracle(q,k,v),atol=3e-12,rtol=3e-12)
    assert actual[:,0].abs().max()<1e-12
    same=torch.ones_like(v)*3
    assert value_moments(q,k,same,centered=False,nonlinearity='silu').abs().max()<1e-12
    # Constant values annihilate the gap and gradients, even for nonuniform A.
    q=torch.randn_like(q).requires_grad_();k=torch.randn_like(k).requires_grad_()
    gap=value_moments(q,k,same,centered=False,nonlinearity='silu')
    assert gap.abs().max()<3e-12
    for grad in torch.autograd.grad(gap.sum(),(q,k)):assert grad.abs().max()<3e-12


def test_initialization_and_optimizer():
    shared.test_matched_initialization_zero_branch_and_optimizer('jensen_silu')


def test_live_2k_causality_and_cached_decode():
    shared.test_live_2k_causality_and_cache('jensen_silu')


def test_compiled_full_model_gradient_and_recompute():
    shared.test_live_compiled_loss_gradients_and_recompute('jensen_silu')


def test_exact_optimizer_and_rng_restore(tmp_path):
    shared.test_checkpoint_optimizer_rng_and_same_gradient_next_update('jensen_silu',tmp_path)


def test_config_roundtrip_and_old_path_values():
    from dataclasses import asdict
    from nanochat.connection_model import ConnectionConfig
    from nanochat.connection_cost import model_cost
    c=ConnectionConfig(attention_statistic='jensen_silu')
    assert ConnectionConfig.from_checkpoint(asdict(c))==c
    assert model_cost(c)==model_cost(ConnectionConfig(attention_statistic='variance'))
    # The default path continues to implement square moments, never SiLU.
    xs=[torch.randn(1,9,1,8,device='cuda',dtype=torch.float64) for _ in range(3)]
    for centered in (False,True):
        torch.testing.assert_close(value_moments(*xs,centered=centered),shared.oracle(*xs,centered),atol=3e-12,rtol=3e-12)
