"""Independent head oracles and the actual compiled training/inference paths."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for
from nanochat.connection_experiment import configure_attention
from nanochat.nonlinear_lmhead import NonlinearLMHeadFeatures
from nanochat.packed_training import seed_all
from nanochat.gpt import norm
from nanochat.engine import KVCache
import test_connection_training_resume as resume_checks

MODES = ('nonlinear_lmhead', 'nonlinear_residual_lmhead')


def record(name, data):
    folder = Path('results/nonlinear-checks'); folder.mkdir(parents=True, exist_ok=True)
    (folder/(name+'.json')).write_text(json.dumps(data, indent=2)+'\n')


def relative(a, b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()


@pytest.mark.parametrize('residual', [False, True])
def test_fp64_formula_and_all_gradients(residual):
    torch.manual_seed(73)
    module = NonlinearLMHeadFeatures(8, residual).double()
    h = torch.randn(3,5,8,dtype=torch.float64,requires_grad=True)
    w = torch.randn(32,8,dtype=torch.float64,requires_grad=True)
    target = torch.randint(0,32,(15,))
    # Independent formula: explicit RMS and sigmoid, not the implementation's
    # norm, silu, Linear, or module forward.
    u = h / (h.square().mean(-1,keepdim=True)+torch.finfo(h.dtype).eps).sqrt() if residual else h
    gate = u @ module.gate.weight.T
    features = gate*torch.sigmoid(gate)*(u @ module.up.weight.T)
    if residual: features = h+features
    expected_logits = features @ w.T
    actual_logits = F.linear(module(h),w)
    expected = F.cross_entropy((15*torch.tanh(expected_logits/15)).flatten(0,1),target)
    actual = F.cross_entropy((15*torch.tanh(actual_logits/15)).flatten(0,1),target)
    variables = (h,module.gate.weight,module.up.weight,w)
    ga = torch.autograd.grad(actual,variables,retain_graph=True)
    gr = torch.autograd.grad(expected,variables)
    torch.testing.assert_close(actual_logits,expected_logits,atol=2e-12,rtol=2e-12)
    for a,b in zip(ga,gr):
        torch.testing.assert_close(a,b,atol=2e-12,rtol=2e-12)
        assert torch.isfinite(a).all() and a.norm()>0
    assert set(dict(module.named_parameters()))=={'gate.weight','up.weight'}
    assert torch.autograd.gradcheck(module,(h,),eps=1e-6,atol=1e-5,rtol=1e-4)
    record('fp64-'+str(residual),dict(logit_max_error=(actual_logits-expected_logits).abs().max().item(),
        all_gradient_relative_errors=[relative(a,b) for a,b in zip(ga,gr)],finite_difference_passed=True,no_alpha=True))


@pytest.mark.parametrize('residual', [False, True])
def test_branch_zero_and_token_isolation(residual):
    module = NonlinearLMHeadFeatures(8,residual).double()
    h = torch.randn(2,4,8,dtype=torch.float64,requires_grad=True)
    gradient = torch.autograd.grad(module(h)[0,1].sum(),h)[0]
    assert gradient[0,1].norm()>0
    gradient[0,1]=0
    assert torch.count_nonzero(gradient)==0
    with torch.no_grad(): module.up.weight.zero_()
    torch.testing.assert_close(module(h),h if residual else torch.zeros_like(h),atol=0,rtol=0)


def build(mode,**kwargs):
    seed_all(42)
    return ConnectionModel(ConnectionConfig(variant='baseline',lm_head_transform=mode,**kwargs)).prepare()


def test_initialization_parameter_ownership_and_old_config():
    models={m:build(m) for m in ('linear',*MODES)}
    reference=models['linear'].state_dict()
    counts={};groups=[];rng=[]
    for name,m in models.items():
        for key,value in reference.items(): assert torch.equal(value,m.state_dict()[key]),key
        assert ConnectionConfig.from_checkpoint(asdict(m.config))==m.config
        counts[name]=sum(p.numel() for p in m.parameters())
        opt,wd=optimizer_for(m)
        groups.append(([{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups],wd))
        ownership={id(p):g['kind'] for g in opt.param_groups for p in g['params']}
        assert ownership[id(m.lm_head.weight)]=='adamw'
        if m.lm_head_features is not None:
            assert all(ownership[id(p)]=='muon' for p in m.lm_head_features.parameters())
        seed_all(42)
        ConnectionModel(m.config)
        rng.append(torch.get_rng_state())
    assert counts==dict(linear=58720256,nonlinear_lmhead=59244544,nonlinear_residual_lmhead=59244544),counts
    assert groups[0]==groups[1]==groups[2]
    assert all(torch.equal(rng[0],r) for r in rng)
    left,right=[models[m].lm_head_features.state_dict() for m in MODES]
    assert all(torch.equal(v,right[k]) for k,v in left.items())
    old=asdict(models['linear'].config);del old['lm_head_transform']
    assert ConnectionConfig.from_checkpoint(old).lm_head_transform=='linear'
    record('initialization',dict(shared_parameters_bitwise_equal=True,gate_up_equal=True,
        rng_unchanged=True,parameters=counts,optimizer_hyperparameters_equal=True,gate_up_optimizer='muon'))


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('scale', [.1,1.])
def test_cce_loss_and_all_head_gradients(mode,scale):
    from cut_cross_entropy import linear_cross_entropy
    seed_all(123)
    module=NonlinearLMHeadFeatures(128,mode==MODES[1]).cuda()
    h=(torch.randn(256,128,device='cuda')*scale).bfloat16().requires_grad_()
    w=(torch.randn(1024,128,device='cuda')*.1).requires_grad_()
    targets=torch.randint(0,1024,(256,),device='cuda')
    f=module(h);cast_w=w.to(f.dtype)
    actual=linear_cross_entropy(f,cast_w,targets,softcap=15.,shift=0,reduction='mean',
        filter_eps=None,filter_e_grad=False,filter_c_grad=False,accum_e_fp32=True,accum_c_fp32=True)
    variables=(h,module.gate.weight,module.up.weight,w)
    ga=torch.autograd.grad(actual,variables,retain_graph=True)
    # Full-vocabulary FP32 reference from the SAME BF16 head features/weights;
    # independent FP64 tests above cover the feature transformation itself.
    logits=F.linear(f.float(),cast_w.float())
    ref=F.cross_entropy(15*torch.tanh(logits/15),targets)
    gr=torch.autograd.grad(ref,variables)
    rounded_ref=F.cross_entropy(15*torch.tanh(logits.detach().to(f.dtype).float()/15),targets)
    errors=[relative(a,b) for a,b in zip(ga,gr)]
    torch.testing.assert_close(actual,rounded_ref,atol=3e-5,rtol=3e-5)
    assert max(errors)<.012,errors
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in ga)
    record(f'cce-{mode}-{scale}',dict(cce=actual.item(),matched_reference=rounded_ref.item(),
        fp32_reference=ref.item(),gradient_relative_errors=errors))


@pytest.mark.parametrize('mode', MODES)
def test_real_2k_causality_cache_compile_and_recompute(mode):
    configure_attention('sdpa_cudnn');torch.backends.cuda.matmul.allow_tf32=False
    m=build(mode,depth=2,width=128,heads=2,vocab_size=256)
    with torch.no_grad():
        for b in m.blocks:
            b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
    ids=torch.randint(0,256,(1,2048),device='cuda')
    changed=ids.clone();changed[:,1024:]=(changed[:,1024:]+31)%256
    with torch.no_grad():
        a,b=m(ids),m(changed)
        torch.testing.assert_close(a[:,:1024],b[:,:1024],atol=0,rtol=0)
        cache=KVCache(1,2,32,64,2,'cuda',torch.bfloat16)
        pre=m(ids[:,:7],kv_cache=cache);one=m(ids[:,7:8],kv_cache=cache);full=m(ids[:,:8])
        torch.testing.assert_close(pre,full[:,:7],atol=.004,rtol=.03)
        torch.testing.assert_close(one,full[:,7:],atol=.004,rtol=.03)
    x,y=ids[:,:128].contiguous(),ids[:,1:129].contiguous()
    eager=m(x,y);eager.backward()
    eager_grads={name:p.grad.clone() for name,p in m.named_parameters()}
    m.zero_grad(set_to_none=True);m.prepare(compile_blocks=True)
    loss=m(x,y);dense=m(x,y,dense=True);loss.backward()
    torch.testing.assert_close(loss,dense,atol=2e-4,rtol=2e-4)
    torch.testing.assert_close(loss,eager,atol=2e-4,rtol=2e-4)
    grads={name:p.grad.clone() for name,p in m.named_parameters()}
    compiled_errors={n:relative(g,eager_grads[n]) for n,g in grads.items()}
    m.zero_grad(set_to_none=True);m.config.activation_checkpoint=True
    again=m(x,y);again.backward()
    errors={name:relative(p.grad,grads[name]) for name,p in m.named_parameters()}
    assert max(compiled_errors.values())<.03,compiled_errors
    assert max(errors.values())<.02,errors
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    record('integration-'+mode,dict(causal_tokens=2048,compiled_gradient_errors=compiled_errors,
        recompute_gradient_errors=errors,cce=loss.item(),dense=dense.item()))


@pytest.mark.parametrize('mode', MODES)
def test_exact_optimizer_rng_resume(mode,tmp_path,monkeypatch):
    monkeypatch.setattr(resume_checks,'build',lambda ignored:build(mode,depth=2,width=128,heads=1,vocab_size=256,sequence_length=128))
    resume_checks.test_optimizer_rng_checkpoint_and_fixed_gradient_next_update('baseline',tmp_path)


def test_old_linear_path_and_input_definition():
    m=build('linear',depth=2,width=128,heads=2,vocab_size=256,sequence_length=128)
    idx=torch.randint(0,256,(2,32),device='cuda')
    with torch.no_grad():
        x=norm(m.wte(idx))
        for block in m.blocks:x=block(x,m.cos[:,:32],m.sin[:,:32])
        h=norm(x)
        torch.testing.assert_close(m.hidden(idx),h,atol=0,rtol=0)
        expected=15*torch.tanh(F.linear(h,m.lm_head.weight.to(h.dtype)).float()/15)
        torch.testing.assert_close(m(idx),expected,atol=0,rtol=0)
    assert m.lm_head_features is None and m.compiled_head_features is None


def test_invalid_combinations():
    with pytest.raises(ValueError):ConnectionConfig(lm_head_transform='unknown')
    for mode in MODES:
        with pytest.raises(ValueError):ConnectionConfig(lm_head_transform=mode,architecture='native')
        with pytest.raises(ValueError):ConnectionConfig(lm_head_transform=mode,variant='mhc',readout='vocab_partition')
