"""GPU-only numerical, routing, causal, compile, and checkpoint checks."""
from dataclasses import asdict

import pytest
import torch

from nanochat.ffn_readout import readout_reference, initialize_gradmax
from nanochat.ffn_readout_triton import fused_readout
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.connection_experiment import configure_attention
from nanochat.packed_training import seed_all, snapshot, restore_rng


def close(a, b, tolerance=.005):
    a,b=a.float(),b.float()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    rel=(a-b).norm()/b.norm().clamp_min(1e-12)
    peak=(a-b).abs().max()/b.abs().max().clamp_min(1e-12)
    assert rel <= tolerance and peak <= 4*tolerance, (rel.item(),peak.item())


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('shape',[(1,128,32),(127,512,64),(4096,512,128),(37,192,48)])
@pytest.mark.parametrize('scale',[1.,.01,.0001])
def test_kernel_forward_backward(dtype,shape,scale):
    seed_all(42)
    m,d,r=shape
    x=(scale*torch.randn(m,d,device='cuda')).to(dtype).requires_grad_()
    p=(scale*torch.randn(m,d+r,device='cuda')).to(dtype).requires_grad_()
    g=torch.randn_like(p)
    ref=readout_reference(x,p)
    ref_grads=torch.autograd.grad(ref,(x,p),g)
    actual=fused_readout(x,p)
    grads=torch.autograd.grad(actual,(x,p),g)
    tol=.005 if dtype==torch.bfloat16 else 3e-6
    close(actual,ref,tol)
    for a,b in zip(grads,ref_grads):close(a,b,tol)


def build(width=32,backend='triton',compiled=False,checkpoint=False):
    seed_all(42)
    config=ConnectionConfig(depth=2,width=128,heads=1,sequence_length=64,vocab_size=512,
        streams=1,ffn_readout_width=width,ffn_readout_backend=backend,activation_checkpoint=checkpoint)
    return ConnectionModel(config).prepare(compile_blocks=compiled)


def test_baseline_initialization_rng_and_new_gradient_path():
    configure_attention('fa2')
    baseline=build(0); baseline_rng=torch.get_rng_state().clone()
    branch=build(); assert torch.equal(baseline_rng,torch.get_rng_state())
    for name,p in baseline.named_parameters():
        other=dict(branch.named_parameters())[name]
        if name=='lm_head.weight':other=other[:,:128]
        if name=='blocks.1.mlp.c_proj.weight':other=other[:128]
        assert torch.equal(p,other),name
    x=torch.randint(0,512,(2,64),device='cuda'); y=(x+1)%512
    close(branch(x),baseline(x),.003)
    branch(x,y,dense=True).backward()
    assert branch.blocks[-1].mlp.c_proj.weight.grad[128:].norm()>0
    assert branch.lm_head.weight.grad[:,128:].count_nonzero()==0
    # A zero conventional head cannot suppress the independent branch's signal.
    branch.zero_grad(set_to_none=True)
    with torch.no_grad():branch.lm_head.weight[:,:128].zero_()
    branch(x,y,dense=True).backward()
    assert branch.blocks[-1].mlp.c_proj.weight.grad[:128].count_nonzero()==0
    assert branch.blocks[-1].mlp.c_proj.weight.grad[128:].norm()>0


def test_shared_gradient_storage_matches_separate_outputs():
    x=torch.randn(128,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    p=torch.randn(128,576,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    g=torch.randn_like(p)
    shared=torch.autograd.grad(fused_readout(x,p,True),(x,p),g)
    separate=torch.autograd.grad(fused_readout(x,p,False),(x,p),g)
    for a,b in zip(shared,separate):assert torch.equal(a,b)
    assert shared[0].untyped_storage().data_ptr()==shared[1].untyped_storage().data_ptr()


@pytest.mark.parametrize('compiled,checkpoint',[(False,False),(True,False),(True,True)])
def test_model_reference_gradients_and_cce(compiled,checkpoint):
    configure_attention('fa2');torch.backends.cuda.matmul.allow_tf32=False
    ref=build(backend='reference',compiled=compiled,checkpoint=checkpoint)
    actual=build(compiled=compiled,checkpoint=checkpoint)
    with torch.no_grad():
        for b in ref.blocks:
            b.mlp.c_proj.weight.normal_(0,.01)
            b.attn.c_proj.weight.normal_(0,.01)
    actual.load_state_dict(ref.state_dict())
    x=torch.randint(0,512,(2,64),device='cuda');y=(x+1)%512
    loss=ref(x,y,dense=True); loss.backward()
    got=actual(x,y);got.backward()
    torch.testing.assert_close(got,loss,rtol=1e-4,atol=2e-5)
    for (name,p),(_,q) in zip(actual.named_parameters(),ref.named_parameters()):
        assert p.grad is not None and q.grad is not None,name
        try:close(p.grad,q.grad,.02)
        except AssertionError as e:raise AssertionError(name,str(e)) from e


def test_gradmax_preserves_function_rng_and_selects_new_directions():
    configure_attention('fa2')
    model=build()
    x=torch.randint(0,512,(2,64),device='cuda');y=(x+1)%512
    with torch.no_grad():before=model(x).clone()
    rng_cpu=torch.get_rng_state().clone();rng_cuda=torch.cuda.get_rng_state().clone()
    report=initialize_gradmax(model,x,y,max_tokens=128)
    assert report['relative_overlap']<1e-4 and report['last_singular_value']>0
    assert torch.equal(rng_cpu,torch.get_rng_state()) and torch.equal(rng_cuda,torch.cuda.get_rng_state())
    with torch.no_grad():torch.testing.assert_close(model(x),before,rtol=0,atol=0)
    model(x,y).backward()
    assert model.blocks[-1].mlp.c_proj.weight.grad[128:].norm()>0


def test_causal_cache_after_nonzero_branch():
    from nanochat import flash_attention
    from nanochat.engine import KVCache
    # SDPA's math path handles the explicit rectangular cached-prefill mask.
    flash_attention.USE_FA3=False
    torch.backends.cuda.enable_math_sdp(True)
    model=build().eval()
    with torch.no_grad():
        for b in model.blocks:
            b.attn.c_proj.weight.normal_(0,.01);b.mlp.c_proj.weight.normal_(0,.01)
        x=torch.randint(0,512,(1,64),device='cuda'); changed=x.clone();changed[:,31:]=(changed[:,31:]+7)%512
        full=model(x)
        close(model(changed)[:,:31],full[:,:31],.003)
        cache=KVCache(1,1,64,128,2,'cuda',torch.bfloat16)
        parts=[];start=0
        for size in (7,13,1,43):
            parts.append(model(x[:,start:start+size],kv_cache=cache));start+=size
        close(torch.cat(parts,1),full,.008)
        assert cache.get_pos()==64


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal_tree(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    else:assert a==b


def test_checkpoint_optimizer_and_next_update(tmp_path):
    configure_attention('fa2')
    model=build();opt,wd=optimizer_for(model)
    x=torch.randint(0,512,(2,64),device='cuda');y=(x+1)%512
    for step in range(3):
        opt.zero_grad(set_to_none=True);schedule(opt,step,100,wd)
        loss=model(x,y,dense=True);loss.backward();opt.step()
    state=snapshot(model,opt,3,config=asdict(model.config),next_data_row=6)
    path=tmp_path/'resume.pt';torch.save(state,path)
    restored=torch.load(path,map_location='cpu',weights_only=False)
    other=ConnectionModel(ConnectionConfig.from_checkpoint(restored['config'])).prepare()
    other_opt,other_wd=optimizer_for(other)
    other.load_state_dict(restored['model']);other_opt.load_state_dict(restored['optimizer'])
    equal_tree(model.state_dict(),other.state_dict());equal_tree(opt.state_dict(),other_opt.state_dict())
    with torch.no_grad():
        assert torch.equal(model.hidden(x),other.hidden(x))
    assert restored['step']==3 and restored['next_data_row']==6
    restore_rng(state['rng']);expected=(torch.rand(4),torch.rand(4,device='cuda'))
    restore_rng(restored['rng']);equal_tree(expected,(torch.rand(4),torch.rand(4,device='cuda')))
    for m,o,w in [(model,opt,wd),(other,other_opt,other_wd)]:
        o.zero_grad(set_to_none=True);schedule(o,3,100,w);m(x,y,dense=True).backward();o.step()
    equal_tree(model.state_dict(),other.state_dict());equal_tree(opt.state_dict(),other_opt.state_dict())


def test_invalid_and_retired_config():
    for kw in [dict(ffn_readout_width=-1),dict(ffn_readout_width=64,streams=4),dict(ffn_readout_backend='bad')]:
        with pytest.raises(ValueError):ConnectionConfig(**kw)
    with pytest.raises(ValueError,match='retired HCA'):
        ConnectionConfig.from_checkpoint(dict(attention_mode='dynamic_hca'))
    assert ConnectionConfig.from_checkpoint(dict(attention_mode='full',hca_block_size=16)).ffn_readout_width==0
