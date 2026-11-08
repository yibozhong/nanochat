"""Dynamic all-site HC: FP64 VJPs, normalized gate paths and model checks."""
from dataclasses import asdict
import json
from pathlib import Path
import pytest
import torch
import test_scaled_hc as base
from nanochat.dynamic_hc import dynamic_read,dynamic_write,reference_read,reference_write,reference_gate
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for
from nanochat.connection_experiment import configure_attention,match_vanilla_initialization
from nanochat.packed_training import seed_all
from test_connection_training_resume import test_optimizer_rng_checkpoint_and_fixed_gradient_next_update as resume_check

CASES=[(1,5,1.,'random'),(131,128,1.,'random'),(129,512,1.,'random'),
       (17,128,0.,'random'),(17,128,1e-4,'random'),(17,128,1.,'zero'),
       (17,128,1.,'saturated'),(32768,512,1.,'random')]
SUFFIXES=('.read_a','.read_b','.mix_a','.mix_b','.post_a','.post_b','.gate_a','.gate_b')


@pytest.fixture(autouse=True)
def clean_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def record(name,x):
    root=Path('results/dynamic-hc-all-checks');root.mkdir(exist_ok=True,parents=True)
    (root/(name+'.json')).write_text(json.dumps(x,indent=2)+'\n')


def inputs(n,c,amp=1.,mode='random'):
    x,y,p,h,_=base.inputs(n,c,amp)
    y=(y.detach()*amp).requires_grad_()
    params=[]
    for _ in range(4):
        v=torch.randn(c,4,device='cuda')*.3
        if mode=='zero':v.zero_()
        if mode=='saturated':v*=40
        params.append(v.requires_grad_())
    return [x,y,p,h,*params]


def check(fn,ref,args,name):
    out=fn(*args);go=torch.randn_like(out);grad=torch.autograd.grad(out,args,go)
    oracle=[v.detach().double().requires_grad_() for v in args]
    exp=ref(*oracle);eg=torch.autograd.grad(exp,oracle,go.double())
    errors=dict(output=base.relative(out,exp),gradients=[base.relative(g,e) for g,e in zip(grad,eg)])
    record(name,errors)
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    assert all(torch.isfinite(g).all() for g in grad)
    again=torch.autograd.grad(fn(*args),args,go)
    assert all(torch.equal(a,b) for a,b in zip(grad,again))


@pytest.mark.parametrize('n,c,amp,mode',CASES)
def test_read_all_vjps(n,c,amp,mode):
    torch.manual_seed(814);torch.backends.cuda.matmul.allow_tf32=False
    x,y,p,h,a,b,pa,pb=inputs(n,c,amp,mode)
    check(dynamic_read,reference_read,[x,p,a,b],f'read-{n}-{c}-{amp}-{mode}')


@pytest.mark.parametrize('n,c,amp,mode',CASES)
def test_write_all_vjps(n,c,amp,mode):
    torch.manual_seed(815);torch.backends.cuda.matmul.allow_tf32=False
    check(dynamic_write,reference_write,inputs(n,c,amp,mode),f'write-{n}-{c}-{amp}-{mode}')


@pytest.mark.parametrize('kind',['read','write'])
def test_compiled_saved_inputs(kind):
    args=inputs(129,128);fn=dynamic_write
    if kind=='read':args=[args[0],args[2],args[4],args[5]];fn=dynamic_read
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda x:(saved.append(x.data_ptr()) or x),lambda x:x):out=fn(*args)
    assert saved==[x.data_ptr() for x in args]
    go=torch.randn_like(out);eg=torch.autograd.grad(out,args,go)
    other=torch.compile(fn,fullgraph=True)(*args);ag=torch.autograd.grad(other,args,go)
    assert torch.equal(out,other)
    assert max(base.relative(a,b) for a,b in zip(ag,eg))<1e-6


def test_token_and_stream_locality():
    x,y,p,h,a,b,pa,pb=inputs(3,128)
    g=reference_gate(x.float(),a,b);changed=x.detach().clone();changed[0,1,7,1]+=3
    g2=reference_gate(changed.float(),a,b)
    # The source's RMS couples its channels; other streams/tokens remain separate.
    mask=torch.ones_like(g,dtype=torch.bool);mask[0,1,:,1]=False
    assert torch.equal(g[mask],g2[mask]) and not torch.equal(g[0,1,:,1],g2[0,1,:,1])
    before=dynamic_write(x,y,p,h,a,b,pa,pb);after=dynamic_write(changed,y,p,h,a,b,pa,pb)
    assert torch.equal(before[:,0],after[:,0]) and torch.equal(before[:,2],after[:,2])
    p2=pa.detach().clone();p2[7,1]+=1
    other=dynamic_write(x,y,p,h,a,b,p2,pb)
    mask=torch.ones_like(before,dtype=torch.bool);mask[:,:,7,1]=False
    assert torch.equal(before[mask],other[mask])


def model():
    configure_attention('hardware_auto');seed_all(42)
    m=ConnectionModel(ConnectionConfig(variant='dynamic_hc_all',width=128,heads=2,depth=2,vocab_size=128)).prepare(compile_blocks=True)
    with torch.no_grad():
        for block in m.blocks:
            block.attn.c_proj.weight.normal_(std=.03);block.mlp.c_proj.weight.normal_(std=.03)
        for name,p in m.named_parameters():
            if name.endswith(SUFFIXES):p.normal_(std=.1)
    return m


def test_identity_rng_shapes_optimizer():
    old=None
    for variant in ('mhc_no_sinkhorn','dynamic_hc_all'):
        seed_all(42)
        m=ConnectionModel(ConnectionConfig(variant=variant,depth=2,width=128,heads=2,vocab_size=128)).prepare()
        match_vanilla_initialization(m,42)
        if old is None:old=m;rng=torch.get_rng_state();continue
        assert torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(p,m.state_dict()[n]) for n,p in old.state_dict().items())
        extra={n:p for n,p in m.named_parameters() if n.endswith(SUFFIXES)}
        assert len(extra)==26 and all(torch.count_nonzero(p)==0 for p in extra.values())
        opt,_=optimizer_for(m);adam={id(p) for g in opt.param_groups if g['kind']=='adamw' and g['weight_decay']==0 for p in g['params']}
        assert all(id(p) in adam for p in extra.values())
        x=torch.randn(2,37,128,4,device='cuda',dtype=torch.bfloat16)
        v,_,r=old.blocks[0].attn_connection.read(x);w,_,s=m.blocks[0].attn_connection.read(x)
        assert torch.equal(r[0],s[0]) and torch.equal(r[1],s[1])
        torch.testing.assert_close(v,w,atol=.0078125,rtol=.008)
        ids=torch.randint(0,128,(2,128),device='cuda')
        with torch.no_grad():torch.testing.assert_close(old(ids),m(ids),atol=2e-5,rtol=2e-5)
        assert ConnectionConfig.from_checkpoint(asdict(m.config))==m.config
    big=ConnectionModel(ConnectionConfig(variant='dynamic_hc_all'))
    assert sum(p.numel() for p in big.parameters())==59716021
    assert sum(p.numel() for n,p in big.named_parameters() if n.endswith(SUFFIXES))==200704


def test_fullmodel_causal_cache_cce_recompute(monkeypatch):
    monkeypatch.setattr(base,'model',model)
    base.test_fullmodel_causal_cache_cce_recompute(monkeypatch)


def test_fullmodel_reference_gradients(monkeypatch):
    m=model();x=torch.randint(0,128,(2,128),device='cuda');y=(x+1)%128
    loss=m(x,y);loss.backward();grads={n:p.grad.clone() for n,p in m.named_parameters()}
    import nanochat.dynamic_hc as module
    monkeypatch.setattr(module,'dynamic_read',lambda x,p,a,b:reference_read(x.float(),p,a,b).to(x.dtype))
    monkeypatch.setattr(module,'dynamic_write',lambda x,y,p,h,a,b,pa,pb:reference_write(x.float(),y.float(),p,h,a,b,pa,pb).to(x.dtype))
    m.zero_grad(set_to_none=True);other=m(x,y);other.backward()
    errors={n:base.relative(grads[n],p.grad) for n,p in m.named_parameters()}
    record('fullmodel',errors)
    torch.testing.assert_close(loss,other,atol=2e-4,rtol=2e-4)
    assert max(errors.values())<.02,errors


def test_optimizer_rng_restore(tmp_path):
    resume_check('dynamic_hc_all',tmp_path,attention_backend='hardware_auto')


@pytest.mark.parametrize('options',[dict(architecture='native'),dict(fused_connection_grad=True),dict(compact_routing=True)])
def test_unsupported(options):
    with pytest.raises(ValueError):ConnectionConfig(variant='dynamic_hc_all',**options)


def test_actual_dispatch():
    args=inputs(127,128)
    out=dynamic_write(*args);out.sum().backward();torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        dynamic_write(*args).sum().backward();torch.cuda.synchronize()
    names=[e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
    assert any('dynamic_hc_fwd' in n for n in names) and any('dynamic_hc_bwd' in n for n in names),names
    assert any('_scale_partial_reduce' in n for n in names),names
    record('dispatch',names)
