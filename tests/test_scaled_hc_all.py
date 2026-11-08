"""All static HC scales: independent FP64 derivatives and model invariants."""
from dataclasses import asdict
import json
from pathlib import Path
import pytest
import torch
import test_scaled_hc as base
from nanochat.scaled_aggregate import scaled_aggregate
from nanochat.scaled_mixing import scaled_mix
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for
from nanochat.connection_experiment import match_vanilla_initialization
from nanochat.packed_training import seed_all
from test_connection_training_resume import test_optimizer_rng_checkpoint_and_fixed_gradient_next_update as resume_check

CASES=[(1,5,1.,'random'),(131,128,1.,'random'),(129,512,1.,'random'),
       (17,128,0.,'random'),(17,128,1e-4,'random'),(17,128,1.,'zero'),
       (17,128,1.,'signed'),(32768,512,1.,'random')]


def record(name,value):
    p=Path('results/scaled-hc-all-checks');p.mkdir(parents=True,exist_ok=True)
    (p/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


def aggregate_reference(x,pre,a):
    return (x*a*pre.unsqueeze(-2)).sum(-1)


def write_reference(x,y,post,h,a,p):
    return torch.einsum('btci,btij->btcj',x*a,h)+y.unsqueeze(-1)*p*post.unsqueeze(-2)


@pytest.fixture(autouse=True)
def fresh_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def check_vjps(fn,ref,args,name):
    out=fn(*args);go=torch.randn_like(out);grads=torch.autograd.grad(out,args,go)
    oracle=[x.detach().double().requires_grad_() for x in args]
    exp=ref(*oracle);eg=torch.autograd.grad(exp,oracle,go.double())
    errors=dict(output=base.relative(out,exp),gradients=[base.relative(g,e) for g,e in zip(grads,eg)])
    record(name,errors)
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    assert all(torch.isfinite(x).all() for x in grads)
    again=torch.autograd.grad(fn(*args),args,go)
    assert all(torch.equal(a,b) for a,b in zip(grads,again))


@pytest.mark.parametrize('n,c,amp,mode',CASES)
def test_read_head_all_vjps(n,c,amp,mode):
    torch.manual_seed(791);torch.backends.cuda.matmul.allow_tf32=False
    args=base.inputs(n,c,amp,mode)
    check_vjps(scaled_aggregate,aggregate_reference,[args[0],args[2],args[4]],f'read-{n}-{c}-{amp}-{mode}')


@pytest.mark.parametrize('n,c,amp,mode',CASES)
def test_post_write_all_vjps(n,c,amp,mode):
    torch.manual_seed(792);torch.backends.cuda.matmul.allow_tf32=False
    args=base.inputs(n,c,amp,mode);p=args[4].detach().clone().requires_grad_()
    check_vjps(scaled_mix,write_reference,[*args,p],f'post-{n}-{c}-{amp}-{mode}')


@pytest.mark.parametrize('kind',['read','write'])
def test_compile_saved_inputs(kind):
    args=base.inputs(129,128)
    fn=scaled_aggregate if kind=='read' else scaled_mix
    args=[args[0],args[2],args[4]] if kind=='read' else [*args,torch.ones_like(args[4],requires_grad=True)]
    saved=[]
    def pack(x):saved.append(x.data_ptr());return x
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):out=fn(*args)
    assert saved==[x.data_ptr() for x in args],saved
    go=torch.randn_like(out);eg=torch.autograd.grad(out,args,go)
    compiled=torch.compile(fn,fullgraph=True)(*args)
    cg=torch.autograd.grad(compiled,args,go)
    assert torch.equal(compiled,out)
    assert max(base.relative(x,y) for x,y in zip(cg,eg))<1e-6


def test_fullmodel_causal_cache_cce_recompute(monkeypatch):
    monkeypatch.setattr(base,'VARIANT','scaled_hc_all')
    base.test_fullmodel_causal_cache_cce_recompute(monkeypatch)


def test_fullmodel_all_reference_gradients(monkeypatch):
    monkeypatch.setattr(base,'VARIANT','scaled_hc_all')
    m=base.model()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if n.endswith(('read_scale','post_scale')) or n=='stream_head.input_scale':p.uniform_(.7,1.3)
    x=torch.randint(0,128,(2,128),device='cuda');y=(x+1)%128
    loss=m(x,y);loss.backward();grads={n:p.grad.clone() for n,p in m.named_parameters()}
    import nanochat.scaled_all_connections as connections
    monkeypatch.setattr(connections,'scaled_aggregate',lambda x,p,a:aggregate_reference(x.float(),p,a).to(x.dtype))
    monkeypatch.setattr(connections,'scaled_mix',lambda x,y,p,h,a,b:write_reference(x.float(),y.float(),p,h,a,b).to(x.dtype))
    m.zero_grad(set_to_none=True);other=m(x,y);other.backward()
    errors={n:base.relative(grads[n],p.grad) for n,p in m.named_parameters()}
    torch.testing.assert_close(loss,other,atol=2e-4,rtol=2e-4)
    assert max(errors.values())<.02,errors
    record('fullmodel-reference',dict(ce=loss.item(),reference=other.item(),gradients=errors))


def test_identity_rng_optimizer_and_shapes():
    old=None
    for variant in ('scaled_hc','scaled_hc_all'):
        seed_all(42)
        m=ConnectionModel(ConnectionConfig(variant=variant,depth=2,width=128,heads=2,vocab_size=128)).prepare()
        match_vanilla_initialization(m,42)
        if old is None:old=m;rng=torch.get_rng_state();continue
        assert torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(p,m.state_dict()[n]) for n,p in old.state_dict().items())
        extra={n:p for n,p in m.named_parameters() if n not in old.state_dict()}
        assert len(extra)==9 and all(torch.equal(p,torch.ones_like(p)) for p in extra.values())
        opt,_=optimizer_for(m);adam={id(p) for g in opt.param_groups if g['kind']=='adamw' and g['weight_decay']==0 for p in g['params']}
        assert all(id(p) in adam for p in extra.values())
        x=torch.randint(0,128,(2,128),device='cuda')
        with torch.no_grad():torch.testing.assert_close(old(x),m(x),atol=2e-5,rtol=2e-5)
        assert ConnectionConfig.from_checkpoint(asdict(m.config))==m.config
    # Source and destination locality of read/post scales.
    x=torch.ones((1,3,128,4),device='cuda',dtype=torch.bfloat16)
    pre=torch.ones((1,3,4),device='cuda');a=torch.ones((128,4),device='cuda')
    before=scaled_aggregate(x,pre,a);a[7,1]=2;after=scaled_aggregate(x,pre,a)
    mask=torch.ones_like(before,dtype=torch.bool);mask[:,:,7]=False
    assert torch.count_nonzero((after-before)[mask])==0 and torch.all((after-before)[:,:,7]==1)
    args=base.inputs(3,128);post_scale=torch.ones((128,4),device='cuda')
    before=scaled_mix(*args,post_scale);post_scale[7,1]=2;after=scaled_mix(*args,post_scale)
    mask=torch.ones_like(before,dtype=torch.bool);mask[:,:,7,1]=False
    assert torch.count_nonzero((after-before)[mask])==0


def test_optimizer_rng_restore(tmp_path):
    resume_check('scaled_hc_all',tmp_path,attention_backend='hardware_auto')


@pytest.mark.parametrize('options',[dict(architecture='native'),dict(fused_connection_grad=True),dict(compact_routing=True)])
def test_unsupported(options):
    with pytest.raises(ValueError):ConnectionConfig(variant='scaled_hc_all',**options)
