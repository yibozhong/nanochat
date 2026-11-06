"""NVIDIA AttnRes: independent all-VJP oracle, causal model, and recovery."""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from nanochat.attention_residuals import _AttnResTritonFunction, AttnResMix, reference_mix
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.engine import KVCache
from nanochat.packed_training import seed_all, snapshot, restore_rng
from test_connection_training_resume import equal_tree


@pytest.fixture(autouse=True)
def isolate_compilation():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def relative(a, b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-12)).item()


def record(name, value):
    root = Path('results/attnres/checks'); root.mkdir(parents=True, exist_ok=True)
    (root/(name+'.json')).write_text(json.dumps(value, indent=2)+'\n')


def oracle(prefix, history, nw, pw):
    # Separate entries; explicit per-entry norm/dot rather than kernel formulas.
    values = list(history.unbind(1)) + [prefix]
    scores = torch.stack([((v/(v.square().mean(-1, keepdim=True)+1e-6).sqrt())*nw*pw).sum(-1)
                          for v in values], -1)
    weights = scores.softmax(-1)
    return sum(v*weights[:, i:i+1] for i, v in enumerate(values))


@pytest.mark.parametrize('n,d,k,amplitude', [
    (1,512,0,1.), (17,512,1,1.), (129,512,4,1.), (67,512,8,1.),
    (19,512,16,1.), (17,128,3,1.), (17,512,8,0.), (17,512,8,1e-4)])
def test_forward_and_every_vjp(n,d,k,amplitude):
    seed_all(101)
    # Non-contiguous token/history strides, contiguous hidden dimension.
    args=[(torch.randn(n*2,d,device='cuda',dtype=torch.bfloat16)[::2]*amplitude).requires_grad_(),
          (torch.randn(n*2,k,d,device='cuda',dtype=torch.bfloat16)[::2]*amplitude).requires_grad_(),
          (1+torch.randn(d,device='cuda')*.1).requires_grad_(),
          (torch.randn(d,device='cuda')*.05).requires_grad_()]
    actual = _AttnResTritonFunction.apply(*args, 1e-6)
    go = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, args, go)
    refs = [p.detach().double().requires_grad_() for p in args]
    expected = oracle(*refs)
    expected_grads = torch.autograd.grad(expected, refs, go.double(), allow_unused=True)
    expected_grads = [torch.zeros_like(p) if g is None else g for p,g in zip(refs,expected_grads)]
    errors = dict(output=relative(actual,expected), gradients=[relative(a,b) for a,b in zip(grads,expected_grads)])
    record(f'oracle-{n}-{d}-{k}-{amplitude}', errors)
    assert errors['output'] < .006, errors
    assert max(errors['gradients']) < .015, errors
    assert all(torch.isfinite(g).all() for g in grads)
    again = torch.autograd.grad(_AttnResTritonFunction.apply(*args,1e-6), args, go)
    assert all(torch.equal(a,b) for a,b in zip(grads,again))


def test_token_locality_and_saved_statistics():
    m=AttnResMix(512).cuda()
    prefix=torch.randn(2,17,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    bank=torch.randn(34,8,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:(saved.append(tuple(t.shape)) or t),lambda t:t):
        out=m(prefix,bank)
    assert saved==[(34,512),(34,8,512),(512,),(512,),(3,34,9)],saved
    changed=bank.detach().clone(); changed[3]+=2
    after=m(prefix,changed)
    mask=torch.ones(34,device='cuda',dtype=torch.bool);mask[3]=False
    assert torch.equal(out.flatten(0,1)[mask],after.flatten(0,1)[mask])
    gp,gb=torch.autograd.grad(out[0,3].float().sum(),(prefix,bank))
    assert torch.count_nonzero(gp.flatten(0,1)[mask])==0 and torch.count_nonzero(gb[mask])==0


@pytest.mark.parametrize('block_size',[1,2,3])
def test_model_block_boundaries_against_independent_reference(block_size):
    configure_attention('hardware_auto'); seed_all(42)
    c=ConnectionConfig(variant='attnres',depth=4,width=128,heads=1,vocab_size=128,attnres_block_size=block_size)
    m=build_connection_model(c).prepare()
    with torch.no_grad():
        for b in m.blocks:
            b.attn.c_proj.weight.normal_(std=.02); b.mlp.c_proj.weight.normal_(std=.02)
    ids=torch.randint(0,128,(2,64),device='cuda')
    from nanochat.gpt import norm
    with torch.no_grad():
        actual=m.hidden(ids)
        prefix=norm(m.wte(ids)); history=[]
        cos,sin=m.cos[:,:64],m.sin[:,:64]
        for i,b in enumerate(m.blocks):
            def mix(reader):
                bank=torch.stack(history,dim=2).flatten(0,1) if history else prefix.new_empty(128,0,128)
                return reference_mix(prefix.flatten(0,1),bank,reader.norm_weight,reader.proj_weight).view_as(prefix)
            h=mix(m.depth_reads[i][0])
            if i%block_size==0:
                history.append(prefix); prefix=None
            a=b.attention_branch(h,cos,sin)
            prefix=a if prefix is None else prefix+a
            prefix=prefix+b.mlp_branch(mix(m.depth_reads[i][1]))
        expected=norm(mix(m.output_attnres))
    error=relative(actual,expected);record(f'boundary-{block_size}',dict(relative=error))
    assert error<.015


def test_full_size_initialization_and_optimizer_ownership():
    configure_attention('hardware_auto')
    seed_all(42);base=build_connection_model(ConnectionConfig()).prepare()
    seed_all(42);m=build_connection_model(ConnectionConfig(variant='attnres')).prepare()
    before={k:v.clone() for k,v in m.state_dict().items()}
    match_vanilla_initialization(m,42)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in base.state_dict().items())
    assert sum(p.numel() for p in m.parameters())==58737664
    opt,_=optimizer_for(m);base_opt,_=optimizer_for(base)
    route_ids={id(p) for p in m.residual_parameters()}
    groups=[g for g in opt.param_groups if any(id(p) in route_ids for p in g['params'])]
    assert len(groups)==1 and groups[0]['kind']=='adamw' and groups[0]['weight_decay']==0
    assert {id(p) for p in groups[0]['params']}==route_ids
    for g in base_opt.param_groups:
        same=[a for a in opt.param_groups if a['kind']==g['kind'] and a['params'][0].shape==g['params'][0].shape
              and a['params'][0].dtype==g['params'][0].dtype]
        assert len(same)==1
        assert {k:v for k,v in g.items() if k!='params'}=={k:v for k,v in same[0].items() if k!='params'}
    assert all(torch.equal(r.norm_weight,torch.ones_like(r.norm_weight)) for pair in m.depth_reads for r in pair)
    assert ConnectionConfig.from_checkpoint(asdict(m.config))==m.config
    for kwargs in (dict(architecture='native'),dict(attnres_block_size=0),dict(depth=17)):
        with pytest.raises(ValueError):ConnectionConfig(variant='attnres',**kwargs)


def model_config():
    return ConnectionConfig(variant='attnres',depth=2,width=128,heads=1,vocab_size=256)


def test_2k_causality_cache_cce_compile_recompute():
    configure_attention('hardware_auto'); seed_all(42)
    m=build_connection_model(model_config()).prepare(compile_blocks=True)
    with torch.no_grad():
        for b in m.blocks:
            b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
    ids=torch.randint(0,256,(2,2048),device='cuda');changed=ids.clone()
    changed[0,1024:]=(changed[0,1024:]+17)%256
    with torch.no_grad():
        a,b=m(ids),m(changed)
        assert torch.equal(a[0,:1024],b[0,:1024]) and torch.equal(a[1],b[1])
        for block in m.blocks:block.compiled_attention=block.compiled_mlp=None
        cache=KVCache(2,1,32,128,2,'cuda',torch.bfloat16)
        pre=m(ids[:,:7],kv_cache=cache);one=m(ids[:,7:8],kv_cache=cache);full=m(ids[:,:8])
        torch.testing.assert_close(pre,full[:,:7],atol=.004,rtol=.03)
        torch.testing.assert_close(one,full[:,7:],atol=.004,rtol=.03)
    x,y=ids[:,:128].contiguous(),ids[:,1:129].contiguous()
    loss=m(x,y);dense=m(x,y,dense=True)
    torch.testing.assert_close(loss,dense,atol=2e-4,rtol=2e-4)
    loss.backward();grads={k:p.grad.clone() for k,p in m.named_parameters()}
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert all(g.norm()>0 for k,g in grads.items() if k.startswith(('depth_reads.','output_attnres.')) and not k.startswith('depth_reads.0.0.'))
    m.zero_grad(set_to_none=True);m.config.activation_checkpoint=True
    again=m(x,y);again.backward()
    errors={k:relative(p.grad,grads[k]) for k,p in m.named_parameters()}
    record('model',dict(ce=loss.item(),dense=dense.item(),recompute_errors=errors))
    assert max(errors.values())<.02,errors


def test_model_gradients_against_torch_reference(monkeypatch):
    configure_attention('hardware_auto');seed_all(31)
    m=build_connection_model(model_config()).prepare()
    with torch.no_grad():
        for b in m.blocks:
            b.attn.c_proj.weight.normal_(std=.02);b.mlp.c_proj.weight.normal_(std=.02)
    x=torch.randint(0,256,(2,128),device='cuda');y=(x+1)%256
    loss=m(x,y);loss.backward();grads={k:p.grad.clone() for k,p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    def reference_forward(self,prefix,history):
        return reference_mix(prefix.flatten(0,1),history,self.norm_weight,self.proj_weight,self.eps).view_as(prefix)
    monkeypatch.setattr(AttnResMix,'forward',reference_forward)
    expected=m(x,y);expected.backward()
    errors={k:relative(grads[k],p.grad) for k,p in m.named_parameters()}
    record('model-reference',dict(loss=loss.item(),reference_loss=expected.item(),gradients=errors))
    assert abs(loss.item()-expected.item())<.001
    assert max(errors.values())<.03,errors


def test_optimizer_rng_and_checkpoint_next_update(tmp_path):
    configure_attention('hardware_auto');seed_all(42)
    c=model_config();m=build_connection_model(c).prepare();opt,wd=optimizer_for(m)
    x=torch.randint(0,256,(2,128),device='cuda');y=(x+1)%256
    for step in range(3):
        opt.zero_grad(set_to_none=True);schedule(opt,step,305176,wd);m(x,y).backward();opt.step()
    grads={k:p.grad.clone() for k,p in m.named_parameters()}
    state=snapshot(m,opt,3,config=asdict(c),next_data_row=48)
    torch.save(state,tmp_path/'latest.pt');state=torch.load(tmp_path/'latest.pt',map_location='cpu',weights_only=False)
    other=build_connection_model(ConnectionConfig.from_checkpoint(state['config'])).prepare()
    oo,_=optimizer_for(other);other.load_state_dict(state['model']);oo.load_state_dict(state['optimizer'])
    equal_tree(m.state_dict(),other.state_dict());equal_tree(opt.state_dict(),oo.state_dict())
    restore_rng(state['rng']);a=torch.rand(8,device='cuda')
    restore_rng(state['rng']);assert torch.equal(a,torch.rand(8,device='cuda'))
    for model,optimizer in ((m,opt),(other,oo)):
        for k,p in model.named_parameters():p.grad=grads[k].clone()
        schedule(optimizer,3,305176,wd);optimizer.step()
    equal_tree(m.state_dict(),other.state_dict());equal_tree(opt.state_dict(),oo.state_dict())


def test_actual_triton_dispatch():
    m=AttnResMix(512).cuda()
    p=torch.randn(1,128,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    b=torch.randn(128,8,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    m(p,b).float().sum().backward()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as trace:
        m(p,b).float().sum().backward()
    keys=[e.key for e in trace.key_averages()]
    for kernel in ('_attn_res_fwd_kernel','_attn_res_bwd_coef_kernel','_attn_res_bwd_grad_kernel'):
        assert any(kernel in k for k in keys),(kernel,keys)
    record('dispatch',dict(operators=keys))
