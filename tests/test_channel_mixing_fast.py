"""Unchanged FP64 thresholds, legacy parity, actual model and restart coverage."""
from dataclasses import asdict
from functools import partial
from pathlib import Path
import json
import pytest
import torch

import test_channel_mixing as original
import test_connection_training_resume as resume
from nanochat.channel_mixing_fast import channel_mix_fast,GEOMETRIES
from nanochat.connection_model import ConnectionConfig,ConnectionModel
from nanochat.connection_experiment import configure_attention,match_vanilla_initialization
from nanochat.packed_training import seed_all
from nanochat.engine import KVCache


def record(name,value):
    root=Path('results/channel-fast-checks');root.mkdir(parents=True,exist_ok=True)
    (root/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


@pytest.fixture(autouse=True)
def fresh_compile_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize('kernel',list(GEOMETRIES))
@pytest.mark.parametrize('n,c,amp',[(1,128,1.),(131,128,1.),(129,512,1.),(17,128,0.),(17,128,1e-4)])
def test_all_vjps_fp64(kernel,n,c,amp,monkeypatch):
    monkeypatch.setattr(original,'channel_mix',partial(channel_mix_fast,kernel=kernel))
    monkeypatch.setattr(original,'record',lambda name,value:record(kernel+'-'+name,value))
    original.test_fused_all_gradients_against_fp64(n,c,amp)


@pytest.mark.parametrize('kernel',list(GEOMETRIES))
def test_locality_and_saved_tensors(kernel,monkeypatch):
    monkeypatch.setattr(original,'channel_mix',partial(channel_mix_fast,kernel=kernel))
    monkeypatch.setattr(original,'record',lambda name,value:record(kernel+'-'+name,value))
    original.test_locality_and_no_saved_coefficient_bank()


@pytest.mark.parametrize('kernel',list(GEOMETRIES))
def test_compiled_primitive(kernel):
    seed_all(51)
    args=[torch.randn(1,129,128,4,device='cuda',dtype=torch.bfloat16,requires_grad=True),
          torch.randn(1,129,128,device='cuda',dtype=torch.bfloat16,requires_grad=True),
          torch.randn(1,129,4,device='cuda',requires_grad=True),
          torch.randn(128,4,4,4,device='cuda',requires_grad=True),
          torch.randn(128,4,4,device='cuda',requires_grad=True),
          torch.tensor([.17],device='cuda',requires_grad=True)]
    fn=partial(channel_mix_fast,kernel=kernel);probe=torch.randn_like(args[0])
    out=fn(*args);grads=torch.autograd.grad(out,args,probe)
    other=torch.compile(fn,fullgraph=True)(*args);gg=torch.autograd.grad(other,args,probe)
    torch.testing.assert_close(other,out,atol=0,rtol=0)
    assert max(original.relative(a,b) for a,b in zip(grads,gg))<1e-6


@pytest.mark.parametrize('kernel',list(GEOMETRIES))
@pytest.mark.parametrize('compact',[False,True])
def test_model_parity_causality_cache_cce_recompute(kernel,compact):
    configure_attention('hardware_auto');torch.backends.cuda.matmul.allow_tf32=False;models=[]
    for k,cp in [('legacy',False),(kernel,compact)]:
        seed_all(42)
        m=ConnectionModel(ConnectionConfig(variant=original.VARIANT,width=128,heads=2,depth=2,
            vocab_size=128,channel_mixing_kernel=k,compact_routing=cp)).prepare(compile_blocks=True)
        match_vanilla_initialization(m,42)
        with torch.no_grad():
            for b in m.blocks:
                b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
        models.append(m)
    old,new=models
    assert all(torch.equal(p,new.state_dict()[k]) for k,p in old.state_dict().items())
    ids=torch.randint(0,128,(2,2048),device='cuda');changed=ids.clone()
    changed[0,1024:]=(changed[0,1024:]+17)%128
    with torch.no_grad():
        a,b=new(ids),new(changed)
        assert torch.equal(a[0,:1024],b[0,:1024]) and torch.equal(a[1],b[1])
        compiled=[(block.compiled_attention,block.compiled_mlp) for block in new.blocks]
        for block in new.blocks:block.compiled_attention=block.compiled_mlp=None
        cache=KVCache(2,2,32,64,2,'cuda',torch.bfloat16)
        pre=new(ids[:,:7],kv_cache=cache);one=new(ids[:,7:8],kv_cache=cache);full=new(ids[:,:8])
        torch.testing.assert_close(pre,full[:,:7],atol=.004,rtol=.03)
        torch.testing.assert_close(one,full[:,7:],atol=.004,rtol=.03)
        for block,(attention,mlp) in zip(new.blocks,compiled):
            block.compiled_attention=attention;block.compiled_mlp=mlp
    assert all(block.compiled_attention is not None and block.compiled_mlp is not None for m in models for block in m.blocks)
    x,y=ids[:,:128].contiguous(),ids[:,1:129].contiguous();grads=[];losses=[]
    for m in models:
        loss=m(x,y);loss.backward();losses.append(loss.detach())
        grads.append({n:p.grad.clone() for n,p in m.named_parameters()})
    errors={n:original.relative(grads[1][n],grads[0][n]) for n in grads[0]}
    record(f'parity-{kernel}-{compact}',dict(gradients=errors,losses=[x.item() for x in losses],both_backbones_compiled=True))
    assert max(errors.values())<(.03 if compact else .02),errors
    assert all(torch.isfinite(g).all() for g in grads[1].values())
    assert all(g.norm()>0 for n,g in grads[1].items() if '_connection.' in n)
    torch.testing.assert_close(losses[0],losses[1],atol=2e-4,rtol=2e-4)
    torch.testing.assert_close(new(x,y,dense=True),losses[1],atol=2e-4,rtol=2e-4)
    new.zero_grad(set_to_none=True);new.config.activation_checkpoint=True
    loss=new(x,y);loss.backward()
    re={n:original.relative(p.grad,grads[1][n]) for n,p in new.named_parameters()}
    assert max(re.values())<.02,re
    record(f'model-{kernel}-{compact}',dict(gradients=errors,recompute=re,losses=[x.item() for x in losses]))


def test_legacy_execution_mode_control():
    """Quantify the previous harness's compiled/eager confound with unchanged HC."""
    configure_attention('hardware_auto');torch.backends.cuda.matmul.allow_tf32=False;models=[]
    for _ in range(2):
        seed_all(42)
        m=ConnectionModel(ConnectionConfig(variant=original.VARIANT,width=128,heads=2,depth=2,
            vocab_size=128)).prepare(compile_blocks=True)
        match_vanilla_initialization(m,42)
        with torch.no_grad():
            for block in m.blocks:
                block.attn.c_proj.weight.normal_(std=.03);block.mlp.c_proj.weight.normal_(std=.03)
        models.append(m)
    assert all(torch.equal(p,models[1].state_dict()[n]) for n,p in models[0].state_dict().items())
    ids=torch.randint(0,128,(2,2048),device='cuda')
    x,y=ids[:,:128].contiguous(),ids[:,1:129].contiguous();grads=[]
    for m in models:
        m(x,y).backward();grads.append({n:p.grad.clone() for n,p in m.named_parameters()})
    matched={n:original.relative(grads[1][n],grads[0][n]) for n in grads[0]}
    other=models[1];other.zero_grad(set_to_none=True)
    for block in other.blocks:block.compiled_attention=block.compiled_mlp=None
    other(x,y).backward()
    unmatched={n:original.relative(p.grad,grads[0][n]) for n,p in other.named_parameters()}
    record('legacy-execution-control',dict(compiled_vs_compiled=matched,compiled_vs_eager=unmatched,
        compiled_vs_compiled_max=max(matched.values()),compiled_vs_eager_max=max(unmatched.values())))
    assert max(matched.values())<.02,matched
    assert all(torch.isfinite(p.grad).all() for p in other.parameters())


@pytest.mark.parametrize('kernel',list(GEOMETRIES))
def test_optimizer_rng_resume(kernel,monkeypatch,tmp_path):
    cls=resume.ConnectionConfig
    monkeypatch.setattr(resume,'ConnectionConfig',partial(cls,channel_mixing_kernel=kernel,compact_routing=True))
    resume.test_optimizer_rng_checkpoint_and_fixed_gradient_next_update(original.VARIANT,tmp_path,attention_backend='hardware_auto')


def test_configuration_compatibility():
    for variant in ('baseline','mhc','mhc_no_sinkhorn','gated_hc','attnres',original.VARIANT):
        assert ConnectionConfig.from_checkpoint(dict(variant=variant)).channel_mixing_kernel=='legacy'
        if variant!=original.VARIANT:
            with pytest.raises(ValueError):ConnectionConfig(variant=variant,channel_mixing_kernel='fused')
    for k in GEOMETRIES:
        c=ConnectionConfig(variant=original.VARIANT,channel_mixing_kernel=k,compact_routing=True)
        assert ConnectionConfig.from_checkpoint(asdict(c))==c
