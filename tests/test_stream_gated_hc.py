"""Independent-stream gates: numerical VJPs, locality and model compatibility."""
from dataclasses import asdict
from pathlib import Path
import json
import pytest
import torch
import nanochat.stream_gated_connections as connections
from nanochat.stream_gate_mixing import stream_gate_mix,stream_gate_reference,project_gates,RMS_EPS
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for
from nanochat.connection_experiment import configure_attention,match_vanilla_initialization
from nanochat.packed_training import seed_all
from nanochat.engine import KVCache
from test_connection_training_resume import test_optimizer_rng_checkpoint_and_fixed_gradient_next_update as resume_check

VARIANT='stream_gated_hc'


def relative(a,b):return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()


def record(name,value):
    p=Path('results/stream-gate-checks');p.mkdir(parents=True,exist_ok=True)
    (p/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


@pytest.fixture(autouse=True)
def fresh_cache():
    torch._dynamo.reset();torch.backends.cuda.matmul.allow_tf32=False
    yield
    torch._dynamo.reset()


def inputs(n,c,amplitude=1.):
    seed_all(19)
    x=(torch.randn(1,n,c,4,device='cuda',dtype=torch.bfloat16)*amplitude).requires_grad_()
    y=torch.randn(1,n,c,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    post=torch.randn(1,n,4,device='cuda',requires_grad=True)
    h=(torch.randn(1,n,4,4,device='cuda')*.2+torch.eye(4,device='cuda')).requires_grad_()
    w=(torch.randn(4,c,c,device='cuda')*.02).requires_grad_()
    return [x,y,post,h,w]


def oracle(x,y,post,h,w):
    # FP64 norm/sigmoid/mixing and GEMM sums, with the production BF16
    # projection boundaries (including their backward cast semantics).
    c=x.shape[-2];n=x.numel()//(4*c)
    u=(x*torch.rsqrt(x.square().mean(-2,keepdim=True)+RMS_EPS)).to(torch.bfloat16).double()
    u=u.reshape(n,c,4).permute(2,0,1).contiguous()
    z=torch.bmm(u,w.to(torch.bfloat16).double().transpose(1,2)).to(torch.bfloat16).double()
    gated=x*torch.sigmoid(z).permute(1,2,0).reshape_as(x)
    return torch.einsum('btci,btij->btcj',gated,h)+y.unsqueeze(-1)*post.unsqueeze(-2)


@pytest.mark.parametrize('n,c,amplitude',[(1,128,1.),(129,128,1.),(131,512,1.),(17,80,1.),(17,128,0.),(17,128,1e-4),(17,128,10.)])
def test_all_vjps_fp64(n,c,amplitude):
    args=inputs(n,c,amplitude);probe=torch.randn_like(args[0])
    out=stream_gate_mix(*args);actual=torch.autograd.grad(out,args,probe)
    refs=[v.detach().double().requires_grad_() for v in args]
    expected_out=oracle(*refs);expected=torch.autograd.grad(expected_out,refs,probe.double())
    errors=dict(output=relative(out,expected_out),gradients=[relative(a,b) for a,b in zip(actual,expected)])
    record(f'vjp-{n}-{c}-{amplitude}',errors)
    assert torch.isfinite(out).all() and all(torch.isfinite(v).all() for v in actual)
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors


def test_stream_token_locality_and_channel_dependence():
    args=inputs(17,128);x,y,p,h,w=args
    z,_=project_gates(x,w);changed=x.detach().clone();changed[0,3,7,1]+=2
    zz,_=project_gates(changed,w)
    mask=torch.ones_like(z,dtype=torch.bool);mask[1,3]=False
    assert torch.equal(z[mask],zz[mask])
    assert torch.count_nonzero(z[1,3]!=zz[1,3])>1
    with torch.no_grad():h.copy_(torch.eye(4,device='cuda').expand_as(h));y.zero_();p.zero_()
    out=stream_gate_mix(*args)
    gx,gw=torch.autograd.grad(out[0,3,9,1],[x,w])
    other=torch.ones_like(x,dtype=torch.bool);other[0,3,:,1]=False
    assert torch.count_nonzero(gx[other])==0 and torch.count_nonzero(gw[[0,2,3]])==0
    assert gx[0,3,:,1].norm()>0 and gw[1].norm()>0
    record('locality',dict(other_tokens_and_streams_independent=True,within_stream_channel_dependence=True))


def test_compiled_primitive_reference():
    args=inputs(129,128);probe=torch.randn_like(args[0])
    out=stream_gate_mix(*args);actual=torch.autograd.grad(out,args,probe)
    ref=torch.compile(stream_gate_reference,fullgraph=True,dynamic=False)(*args)
    expected=torch.autograd.grad(ref,args,probe)
    errors=dict(output=relative(out,ref),gradients=[relative(a,b) for a,b in zip(actual,expected)])
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    record('compiled-primitive',errors)


def build(variant=VARIANT):
    configure_attention('hardware_auto');seed_all(42)
    m=ConnectionModel(ConnectionConfig(variant=variant,width=128,heads=2,depth=2,vocab_size=128)).prepare(compile_blocks=True)
    match_vanilla_initialization(m,42)
    return m


def test_parameters_routing_rng_optimizer():
    old=build('mhc_no_sinkhorn');rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    new=build();assert torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    assert all(torch.equal(p,new.state_dict()[name]) for name,p in old.state_dict().items())
    extra={name for name in new.state_dict() if name not in old.state_dict()}
    assert len(extra)==4 and all(n.endswith('.gate_weight') for n in extra)
    assert all(new.state_dict()[n].shape==(4,128,128) for n in extra)
    a,b=old.blocks[0].attn_connection,new.blocks[0].attn_connection
    x=torch.randn(2,31,128,4,device='cuda',dtype=torch.bfloat16)
    av,_,am=a.read(x);bv,_,bm=b.read(x)
    assert torch.equal(av,bv) and torch.equal(am[0],bm[0]) and torch.equal(am[1],bm[1])
    opt,_=optimizer_for(new);routed={id(p) for g in opt.param_groups if g['kind']=='adamw' and g['weight_decay']==0 for p in g['params']}
    assert all(id(p) in routed for name,p in new.named_parameters() if '_connection.' in name)
    added=sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters())
    assert added==4*128*128*4 and ConnectionConfig.from_checkpoint(asdict(new.config))==new.config
    record('ownership',dict(added_parameters=added,same_rng=True,unchanged_routing=True,independent_weights=True))


def nonzero_model():
    m=build()
    with torch.no_grad():
        for b in m.blocks:b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
    return m


def test_model_causality_cache_cce_recompute():
    m=nonzero_model();ids=torch.randint(0,128,(2,2048),device='cuda');changed=ids.clone();changed[0,1024:]=(changed[0,1024:]+17)%128
    with torch.no_grad():
        a,b=m(ids),m(changed)
        assert torch.equal(a[0,:1024],b[0,:1024]) and torch.equal(a[1],b[1])
        compiled=[(b.compiled_attention,b.compiled_mlp) for b in m.blocks]
        for b in m.blocks:b.compiled_attention=b.compiled_mlp=None
        cache=KVCache(2,2,32,64,2,'cuda',torch.bfloat16)
        pre=m(ids[:,:7],kv_cache=cache);one=m(ids[:,7:8],kv_cache=cache);full=m(ids[:,:8])
        torch.testing.assert_close(pre,full[:,:7],atol=.004,rtol=.03)
        torch.testing.assert_close(one,full[:,7:],atol=.004,rtol=.03)
        for b,(attn,mlp) in zip(m.blocks,compiled):b.compiled_attention=attn;b.compiled_mlp=mlp
    x,y=ids[:,:128].contiguous(),ids[:,1:129].contiguous()
    loss=m(x,y);torch.testing.assert_close(loss,m(x,y,dense=True),atol=2e-4,rtol=2e-4)
    loss.backward();grads={n:p.grad.clone() for n,p in m.named_parameters()}
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert all(g.norm()>0 for n,g in grads.items() if '_connection.' in n)
    m.zero_grad(set_to_none=True);m.config.activation_checkpoint=True;other=m(x,y);other.backward()
    errors={n:relative(p.grad,grads[n]) for n,p in m.named_parameters()}
    assert max(errors.values())<.02,errors
    record('model',dict(loss=loss.item(),gradients=errors))


def test_model_reference(monkeypatch):
    m=nonzero_model();x=torch.randint(0,128,(2,128),device='cuda');y=(x+1)%128
    loss=m(x,y);loss.backward();grads={n:p.grad.clone() for n,p in m.named_parameters()}
    monkeypatch.setattr(connections,'stream_gate_mix',torch.compile(stream_gate_reference,fullgraph=True,dynamic=False))
    m.zero_grad(set_to_none=True);other=m(x,y);other.backward()
    errors={n:relative(p.grad,grads[n]) for n,p in m.named_parameters()}
    assert max(errors.values())<.02,errors
    torch.testing.assert_close(loss,other,atol=2e-4,rtol=2e-4)
    record('model-reference',dict(gradients=errors,loss=loss.item(),reference=other.item()))


def test_optimizer_rng_resume(tmp_path):resume_check(VARIANT,tmp_path,attention_backend='hardware_auto')


@pytest.mark.parametrize('options',[dict(architecture='native'),dict(fused_connection_grad=True),dict(compact_routing=True)])
def test_invalid_config(options):
    with pytest.raises(ValueError):ConnectionConfig(variant=VARIANT,**options)
