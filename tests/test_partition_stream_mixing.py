"""Final K-to-K readout mixing: independent math, information paths and resume."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from nanochat.nvidia_connections import NvidiaStreamMixer, NvidiaHyperConnection
from nanochat.connection_model import ConnectionConfig, optimizer_for
import test_partitioned_readout as partition_checks
import test_connection_training_resume as resume_checks


def record(name, value):
    p=Path('results/partition-mix-checks');p.mkdir(parents=True,exist_ok=True)
    (p/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


@pytest.mark.parametrize('width',[128,512])
def test_final_mixer_fp64_forward_backward_and_layer_equivalence(width):
    torch.manual_seed(724)
    m=NvidiaStreamMixer(width).cuda();m.reset_parameters()
    x=torch.randn(2,64,width,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    actual=m(x);probe=torch.randn_like(actual).float()/actual.numel()
    ga=torch.autograd.grad((actual.float()*probe).sum(),(x,*m.parameters()))
    xr=x.detach().double().requires_grad_()
    phi,bias,scale=[p.detach().double().requires_grad_() for p in m.parameters()]
    flat=xr.flatten(-2)
    raw=F.linear(flat/(flat.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt(),phi)
    logp=(raw*scale+bias).reshape(2,64,4,4)
    for _ in range(20):
        logp=logp-torch.logsumexp(logp,-1,keepdim=True)
        logp=logp-torch.logsumexp(logp,-2,keepdim=True)
    expected=torch.einsum('btcn,btnm->btcm',xr,logp.exp())
    gr=torch.autograd.grad((expected*probe.double()).sum(),(xr,phi,bias,scale))
    relative=partition_checks.relative
    errors=dict(output=relative(actual,expected),gradients=[relative(a,b) for a,b in zip(ga,gr)])
    assert errors['output']<.01 and max(errors['gradients'])<.03,errors
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in ga)
    # Copy exactly the residual coefficients into an ordinary layer connector.
    old=NvidiaHyperConnection(width,4,0).cuda();old.reset_parameters()
    with torch.no_grad():
        old.phi[8:].copy_(m.phi);old.bias[8:].copy_(m.bias);old.scales[2].copy_(m.scale[0])
        _,_,(_,matrix,_)=old.read(x.detach())
        new_matrix=m.mixing_matrix(x.detach())
        torch.testing.assert_close(matrix,new_matrix,atol=1e-6,rtol=1e-5)
        sv=torch.linalg.svdvals(new_matrix.double())
        assert sv[...,-1].min()>.25  # Specific initialized matrices, not a Sinkhorn guarantee.
        common=x.detach()[...,:1].expand_as(x).contiguous()
        torch.testing.assert_close(m(common),common,atol=.016,rtol=.01)
    # One output stream has a differentiable path to every input stream.
    z=x.detach().requires_grad_();out=m(z)
    g=torch.autograd.grad((out[...,0].float()*probe[...,0]).sum(),z)[0]
    norms=g.float().square().sum((0,1,2)).sqrt()
    assert (norms>0).all()
    errors.update(single_output_input_gradient_norms=norms.tolist(),
                  initial_minimum_singular_value=sv[...,-1].min().item(),
                  layer_residual_matrix_equivalent=True)
    record(f'math-d{width}',errors)


def test_fixed_full_rank_mix_preserves_head_capacity_but_sinkhorn_is_not_guarantee():
    torch.manual_seed(8)
    k,d,v=4,5,48
    w=torch.randn(k,v//k,d,dtype=torch.float64)
    head=torch.block_diag(*w)
    matrix=.2*torch.eye(k,dtype=torch.float64)+.8*torch.ones(k,k,dtype=torch.float64)/k
    projected=head@torch.kron(matrix.T.contiguous(),torch.eye(d,dtype=torch.float64))
    assert torch.linalg.matrix_rank(projected)==k*d
    mean=torch.ones(k,k,dtype=torch.float64)/k
    assert torch.linalg.matrix_rank(mean)==1  # Positive, doubly stochastic, yet singular.
    record('rank',dict(full_rank_mix_output_rank=k*d,singular_doubly_stochastic_example_rank=1,
        scope='Fixed coefficient linear readout; excludes RMSNorm/softcap and derivative of dynamic coefficients'))


def build_mixed(readout='vocab_partition',**kwargs):
    return partition_checks.build(readout,partition_mixing='mhc',**kwargs)


def test_shared_initialization_optimizer_and_old_checkpoint_defaults():
    left=partition_checks.build('vocab_partition');right=build_mixed()
    a,b=left.state_dict(),right.state_dict()
    assert set(b)-set(a)=={'partition_mixer.phi','partition_mixer.bias','partition_mixer.scale'}
    assert all(torch.equal(value,b[name]) for name,value in a.items())
    groups=[]
    for model in (left,right):
        opt,wd=optimizer_for(model)
        groups.append(([{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups],wd))
    assert groups[0]==groups[1]
    counts=[sum(p.numel() for p in m.parameters()) for m in (left,right)]
    assert counts==[59507120,59539905],counts
    config=asdict(left.config);del config['partition_mixing']
    assert ConnectionConfig.from_checkpoint(config)==left.config
    left.load_state_dict(a,strict=True)
    assert right.hidden(torch.zeros(1,32,device='cuda',dtype=torch.long)).shape==(1,32,4,512)
    record('initialization',dict(shared_parameters_bitwise_equal=True,
        optimizer_hyperparameters_equal=True,parameters=counts,added_parameters=32785,
        legacy_checkpoint_default='none'))


def test_mixed_readout_causal_cache_compile_cce_and_recompute(monkeypatch):
    original=partition_checks.build
    monkeypatch.setattr(partition_checks,'build',lambda readout,**kw:original(readout,partition_mixing='mhc',**kw))
    partition_checks.test_causal_cache_compile_and_checkpoint()
    value=json.loads(Path('results/partition-checks/integration.json').read_text())
    record('integration',value)


def test_mixed_optimizer_rng_exact_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(resume_checks,'build',lambda variant:build_mixed(
        depth=2,width=128,heads=1,vocab_size=256,sequence_length=128))
    resume_checks.test_optimizer_rng_checkpoint_and_fixed_gradient_next_update('mhc',tmp_path)
    record('resume',dict(model_optimizer_rng_exact=True,fixed_gradient_next_update_exact=True))
