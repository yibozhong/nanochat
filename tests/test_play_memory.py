"""Independent serial VJPs, chunk boundaries, real model and restart checks."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nanochat.play_memory import play_reference
from nanochat.play_memory_kernels import play_relu2
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.connection_experiment import configure_attention
from nanochat.packed_training import seed_all, snapshot, restore_rng
from nanochat.engine import KVCache
from test_connection_training_resume import equal_tree
from test_skip_scale import test_actual_attention_forward_backward_backend


def relative(a,b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-12)).item()


def record(name,value):
    root=Path('results/play-checks');root.mkdir(parents=True,exist_ok=True)
    (root/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


@pytest.fixture(autouse=True)
def reset_compiler():
    torch._dynamo.reset();torch.backends.cuda.matmul.allow_tf32=False
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize('t,h,mode',[(1,7,'random'),(2,33,'random'),(127,17,'random'),
    (128,65,'random'),(129,33,'random'),(257,7,'random'),(2048,13,'random'),
    (131,17,'zero'),(129,7,'tiny'),(257,13,'ties'),(129,33,'identity')])
def test_serial_oracle_all_vjps(t,h,mode):
    seed_all(9)
    u=torch.randn(2,t,h,device='cuda',dtype=torch.bfloat16)
    r=torch.rand(h,device='cuda')*.7+.05
    w=torch.randn(h,device='cuda')*.3
    if mode=='zero':u.zero_()
    if mode=='tiny':u.mul_(1e-5);r.fill_(1e-6)
    if mode=='ties':u.copy_(torch.randint(-2,3,u.shape,device='cuda'));r.fill_(1)
    if mode=='identity':w.zero_()
    args=[v.requires_grad_() for v in (u,r,w)]
    go=torch.randn_like(u)
    oracle=[v.detach().double().requires_grad_() for v in args]
    expected=play_reference(*oracle)
    eg=torch.autograd.grad(expected,oracle,go.double())
    out=play_relu2(*args);grads=torch.autograd.grad(out,args,go)
    errors=dict(output=relative(out,expected),gradients=[relative(a,b) for a,b in zip(grads,eg)])
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    assert out.dtype==u.dtype and grads[1].dtype==grads[2].dtype==torch.float32
    if mode=='identity':assert torch.equal(out,torch.relu(u).square())
    record(f'oracle-{t}-{h}-{mode}',errors)


def test_full_shape_channel_oracle():
    seed_all(10)
    u=torch.randn(16,2048,2048,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    r=torch.full((2048,),.25,device='cuda',requires_grad=True)
    w=torch.full((2048,),.2,device='cuda',requires_grad=True)
    go=torch.randn_like(u)
    out=play_relu2(u,r,w);grads=torch.autograd.grad(out,(u,r,w),go)
    assert torch.isfinite(out).all() and all(torch.isfinite(g).all() for g in grads)
    oracle=[v.detach().double().requires_grad_() for v in (u[:,:,:4],r[:4],w[:4])]
    expected=play_reference(*oracle);eg=torch.autograd.grad(expected,oracle,go[:,:,:4].double())
    errors=dict(output=relative(out[:,:,:4],expected),gradients=[relative(grads[0][:,:,:4],eg[0]),
        relative(grads[1][:4],eg[1]),relative(grads[2][:4],eg[2])])
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    record('production-shape',errors)


def test_compiled_primitive_keeps_inputs_immutable():
    seed_all(17)
    u=torch.randn(2,129,33,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    r=torch.full((33,),.25,device='cuda',requires_grad=True)
    w=torch.full((33,),.2,device='cuda',requires_grad=True)
    args=[u,r,w];saved=[v.detach().clone() for v in args];go=torch.randn_like(u)
    compiled=torch.compile(play_relu2,fullgraph=True,dynamic=False)
    out=compiled(*args);grads=torch.autograd.grad(out,args,go)
    oracle=[v.double().requires_grad_() for v in saved]
    expected=play_reference(*oracle);eg=torch.autograd.grad(expected,oracle,go.double())
    assert all(torch.equal(v,s) for v,s in zip(args,saved))
    errors=dict(output=relative(out,expected),gradients=[relative(a,b) for a,b in zip(grads,eg)])
    assert errors['output']<.006 and max(errors['gradients'])<.012,errors
    record('compiled-primitive-immutable',errors)


def build(memory='play',compiled=False,nonzero=False):
    configure_attention('hardware_auto');seed_all(42)
    model=ConnectionModel(ConnectionConfig(mlp_memory=memory,streams=1,depth=2,width=128,
        heads=2,sequence_length=2048,vocab_size=128)).prepare(compile_blocks=compiled)
    if nonzero:
        with torch.no_grad():
            for block in model.blocks:
                block.attn.c_proj.weight.normal_(std=.03);block.mlp.c_proj.weight.normal_(std=.03)
                if memory=='play':block.mlp.play.memory_weight.fill_(.2)
    return model


def test_initialization_identity_ownership_and_metadata():
    m=build();rng=torch.get_rng_state();base=build('none')
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(p,m.state_dict()[n]) for n,p in base.state_dict().items())
    with torch.no_grad():
        for b in base.blocks:
            b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
        m.load_state_dict(base.state_dict(),strict=False)
        ids=torch.randint(0,128,(2,129),device='cuda')
        assert torch.equal(m(ids),base(ids))
    extra={n:p for n,p in m.named_parameters() if '.play.' in n}
    assert len(extra)==4 and all(p.shape==(512,) and p.dtype==torch.float32 for p in extra.values())
    opt,_=optimizer_for(m)
    groups=[g for g in opt.param_groups if g['kind']=='adamw' and g['weight_decay']==0]
    assert len(groups)==1 and {id(p) for p in groups[0]['params']}=={id(p) for p in extra.values()}
    assert all(p.ndim==2 for g in opt.param_groups if g['kind']=='muon' for p in g['params'])
    assert ConnectionConfig.from_checkpoint(asdict(m.config))==m.config
    old=asdict(base.config);old.pop('mlp_memory');old.pop('play_radius_init')
    assert ConnectionConfig.from_checkpoint(old).mlp_memory=='none'
    with torch.device('meta'):
        full=ConnectionModel(ConnectionConfig(mlp_memory='play',streams=1))
    assert sum(p.numel() for p in full.parameters())==58753024


def test_fullmodel_causality_cce_recompute_and_cache_rejection():
    m=build(compiled=True,nonzero=True)
    ids=torch.randint(0,128,(2,2048),device='cuda');changed=ids.clone()
    changed[0,1024:]=(changed[0,1024:]+17)%128
    with torch.no_grad():
        before,after=m(ids),m(changed)
        assert torch.equal(before[0,:1024],after[0,:1024]) and torch.equal(before[1],after[1])
        assert torch.equal(before,m(ids))  # No cross-call state.
    cache=KVCache(2,2,32,64,2,'cuda',torch.bfloat16)
    with pytest.raises(ValueError,match='state cache'):m(ids[:,:7],kv_cache=cache)
    assert cache.get_pos()==0
    x,y=ids[:,:129].contiguous(),ids[:,1:130].contiguous()
    loss=m(x,y);dense=m(x,y,dense=True)
    torch.testing.assert_close(loss,dense,atol=2e-4,rtol=2e-4)
    loss.backward();grads={n:p.grad.clone() for n,p in m.named_parameters()}
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert all(g.norm()>0 for n,g in grads.items() if '.play.' in n)
    m.zero_grad(set_to_none=True);m.config.activation_checkpoint=True
    again=m(x,y);again.backward()
    errors={n:relative(p.grad,grads[n]) for n,p in m.named_parameters()}
    assert max(errors.values())<.02,errors
    record('fullmodel',dict(loss=loss.item(),recompute_errors=errors))


def test_compiled_model_against_serial_memory():
    reference=build(nonzero=True);compiled=build(compiled=True,nonzero=True)
    equal_tree(reference.state_dict(),compiled.state_dict())
    for b in reference.blocks:
        module=b.mlp.play
        module.forward=lambda u,module=module:play_reference(u.float(),torch.nn.functional.softplus(module.raw_radius),module.memory_weight).to(u.dtype)
    x=torch.randint(0,128,(2,129),device='cuda');y=(x+1)%128
    a=reference(x,y);a.backward();b=compiled(x,y);b.backward()
    torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
    errors={n:relative(p.grad,dict(reference.named_parameters())[n].grad) for n,p in compiled.named_parameters()}
    assert max(errors.values())<.03,errors
    record('serial-model',errors)


def test_optimizer_rng_restore(tmp_path):
    m=build(compiled=True);opt,wd=optimizer_for(m)
    x=torch.randint(0,128,(2,129),device='cuda');y=(x+1)%128
    for step in range(4):
        schedule(opt,step,305176,wd);opt.zero_grad(set_to_none=True);m(x,y).backward();opt.step()
    gradients={n:p.grad.detach().clone() for n,p in m.named_parameters()}
    path=tmp_path/'state.pt';torch.save(snapshot(m,opt,4,config=asdict(m.config)),path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    other=build();other_opt,_=optimizer_for(other)
    other.load_state_dict(state['model']);other_opt.load_state_dict(state['optimizer'])
    equal_tree(m.state_dict(),other.state_dict());equal_tree(opt.state_dict(),other_opt.state_dict())
    restore_rng(state['rng']);expected=(torch.rand(8),torch.rand(8,device='cuda'),np.random.rand(8))
    restore_rng(state['rng']);equal_tree(expected,(torch.rand(8),torch.rand(8,device='cuda'),np.random.rand(8)))
    for target,optimizer in ((m,opt),(other,other_opt)):
        for n,p in target.named_parameters():p.grad=gradients[n].clone()
        schedule(optimizer,4,305176,wd);optimizer.step()
    equal_tree(m.state_dict(),other.state_dict());equal_tree(opt.state_dict(),other_opt.state_dict())


@pytest.mark.parametrize('options',[dict(architecture='native'),dict(variant='mhc'),dict(streams=4),
    dict(play_radius_init=0),dict(sequence_length=16384)])
def test_invalid_configuration(options):
    kwargs=dict(mlp_memory='play',streams=1);kwargs.update(options)
    with pytest.raises(ValueError):ConnectionConfig(**kwargs)
