"""Orthogonal final mixing: solve-based oracle and production GPU gradients."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from nanochat.nvidia_connections import common_preserving_cayley, NvidiaOrthogonalStreamMixer
from nanochat.nvidia_ops import nvidia_ops
from nanochat.connection_model import optimizer_for
import test_partitioned_readout as partition_checks
import test_connection_training_resume as resume_checks


def record(name,value):
    p=Path('results/orthogonal-mix-checks');p.mkdir(parents=True,exist_ok=True)
    (p/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


def reference_rotation(coords):
    a,b,c=coords.unbind(-1);zero=torch.zeros_like(a)
    skew=torch.stack((zero,-c,b,c,zero,-a,-b,a,zero),-1).reshape(*coords.shape[:-1],3,3)
    eye=torch.eye(3,device=coords.device,dtype=coords.dtype)
    rotation=torch.linalg.solve(eye-skew,eye+skew)
    basis=torch.tensor([[1,1,1],[-1,1,-1],[1,-1,-1],[-1,-1,1]],device=coords.device,dtype=coords.dtype)/2
    return .25+basis@rotation@basis.T


@pytest.mark.parametrize('scale',[0.,.01,1.,100.])
def test_cayley_solve_oracle_gradients_and_contrast_norm(scale):
    torch.manual_seed(74)
    coords=(scale*torch.randn(128,3,device='cuda',dtype=torch.float64)).requires_grad_()
    q=common_preserving_cayley(coords);ref=reference_rotation(coords)
    torch.testing.assert_close(q,ref,atol=1e-12,rtol=1e-10)
    probe=torch.randn_like(q)
    a=torch.autograd.grad((q*probe).sum(),coords,retain_graph=True)[0]
    b=torch.autograd.grad((ref*probe).sum(),coords)[0]
    torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-9)
    eye=torch.eye(4,device='cuda',dtype=torch.float64)
    torch.testing.assert_close(q.transpose(-1,-2)@q,eye.expand_as(q),atol=1e-11,rtol=1e-10)
    torch.testing.assert_close(q.sum(-1),torch.ones_like(q[...,0]),atol=1e-11,rtol=1e-10)
    torch.testing.assert_close(q.sum(-2),torch.ones_like(q[...,0]),atol=1e-11,rtol=1e-10)
    h=torch.randn(128,7,4,device='cuda',dtype=torch.float64);out=h@q
    torch.testing.assert_close(out.mean(-1),h.mean(-1),atol=1e-11,rtol=1e-10)
    torch.testing.assert_close((out-out.mean(-1,keepdim=True)).square().sum((-1,-2)),
                              (h-h.mean(-1,keepdim=True)).square().sum((-1,-2)),atol=1e-10,rtol=1e-10)
    record(f'cayley-{scale}',dict(matrix_max_error=(q-ref).abs().max().item(),
        gradient_max_error=(a-b).abs().max().item(),common_and_contrast_preserved=True))


@pytest.mark.parametrize('width',[128,512])
def test_nvidia_orthogonal_mixer_fp64_and_signed_backward(width,monkeypatch):
    torch.manual_seed(47)
    m=NvidiaOrthogonalStreamMixer(width).cuda();m.reset_parameters()
    with torch.no_grad():m.scale.fill_(.4)  # Exercise appreciable signed rotations.
    def forbidden(*args,**kwargs):raise AssertionError('Final orthogonal mixer must not invoke Sinkhorn')
    monkeypatch.setattr(nvidia_ops(),'mhc_fused_sinkhorn',forbidden)
    x=torch.randn(2,64,width,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    out=m(x);probe=torch.randn_like(out).float()/out.numel()
    ga=torch.autograd.grad((out.float()*probe).sum(),(x,*m.parameters()))
    xr=x.detach().double().requires_grad_();phi,bias,scale=[p.detach().double().requires_grad_() for p in m.parameters()]
    flat=xr.flatten(-2)
    coords=F.linear(flat/(flat.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt(),phi)*scale+bias
    ref=xr@reference_rotation(coords)
    gr=torch.autograd.grad((ref*probe.double()).sum(),(xr,phi,bias,scale))
    rel=partition_checks.relative
    errors=dict(output=rel(out,ref),gradients=[rel(a,b) for a,b in zip(ga,gr)])
    assert errors['output']<.01 and max(errors['gradients'])<.03,errors
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in ga)
    with torch.no_grad():
        q=m.mixing_matrix(x.detach()).double()
        error=(q.transpose(-1,-2)@q-torch.eye(4,device='cuda',dtype=torch.float64)).abs().max().item()
        assert error<3e-6,error
        assert (q<0).any()
        common=x.detach()[...,:1].expand_as(x).contiguous()
        torch.testing.assert_close(m(common),common,atol=.016,rtol=.01)
        diff=lambda z:(z.double()-z.double().mean(-1,keepdim=True)).square().sum()
        retention=(diff(out)/diff(x)).item();assert abs(retention-1)<.01,retention
    errors.update(orthogonality_max_error=error,contrast_energy_retention=retention,final_sinkhorn_calls=0)
    record(f'nvidia-d{width}',errors)


def build_orthogonal(readout='vocab_partition',**kwargs):
    return partition_checks.build(readout,partition_mixing='orthogonal',**kwargs)


def test_three_way_initialization_and_optimizer():
    models=[partition_checks.build('vocab_partition',partition_mixing=m) for m in ['none','mhc','orthogonal']]
    base=models[0].state_dict();counts=[];groups=[]
    for m in models:
        state=m.state_dict();assert all(torch.equal(value,state[name]) for name,value in base.items())
        opt,wd=optimizer_for(m)
        groups.append(([{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups],wd))
        counts.append(sum(p.numel() for p in m.parameters()))
    assert counts==[59507120,59539905,59513268],counts
    assert groups[0]==groups[1]==groups[2]
    assert all(c.sinkhorn for c in [b.attn_connection for b in models[2].blocks]+[b.mlp_connection for b in models[2].blocks])
    record('initialization',dict(shared_parameters_bitwise_equal=True,optimizer_hyperparameters_equal=True,
        parameters=dict(zip(['none','mhc','orthogonal'],counts)),orthogonal_added_parameters=6148,
        backbone_sinkhorn_unchanged=True,orthogonal_config=asdict(models[2].config)))


def test_orthogonal_causal_cache_compile_cce_and_recompute(monkeypatch):
    original=partition_checks.build
    monkeypatch.setattr(partition_checks,'build',lambda readout,**kw:original(readout,partition_mixing='orthogonal',**kw))
    partition_checks.test_causal_cache_compile_and_checkpoint()
    record('integration',json.loads(Path('results/partition-checks/integration.json').read_text()))


def test_orthogonal_exact_optimizer_rng_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(resume_checks,'build',lambda variant:build_orthogonal(
        depth=2,width=128,heads=1,vocab_size=256,sequence_length=128))
    resume_checks.test_optimizer_rng_checkpoint_and_fixed_gradient_next_update('mhc',tmp_path)
    record('resume',dict(model_optimizer_rng_exact=True,fixed_gradient_next_update_exact=True))
