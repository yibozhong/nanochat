"""Independent recurrence, gradients, causal execution and checkpoint checks."""
from dataclasses import asdict
import gc

import pytest
import torch
import torch.nn.functional as F

from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.connection_experiment import configure_attention
from nanochat.ffn_width_residual import ffn_width_residual
from nanochat.gpt import norm
from nanochat.packed_training import seed_all, snapshot, restore_rng
from test_connection_training_resume import equal_tree


@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(2)
    seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.recompile_limit = 64
    configure_attention('fa3')
    yield
    gc.collect()
    torch.cuda.empty_cache()


def build(enabled=True, compiled=False, nonzero=True, **kwargs):
    seed_all(42)
    options = dict(depth=3, width=128, heads=1, vocab_size=256, sequence_length=128,
                   ffn_width_residual=enabled)
    options.update(kwargs)
    model = ConnectionModel(ConnectionConfig(**options)).prepare(compile_blocks=compiled)
    # Test actual computation rather than only the initial zero output projections.
    if nonzero:
        with torch.no_grad():
            for b in model.blocks:
                b.attn.c_proj.weight.normal_(std=.02)
                b.mlp.c_proj.weight.normal_(std=.02)
    return model


def relative(a, b):
    return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-9))


def test_formula_and_gradients_double():
    z, s = [torch.randn(2, 3, 8, device='cuda', dtype=torch.float64,
                        requires_grad=True) for _ in range(2)]
    a, b = [x.detach().clone().requires_grad_() for x in (z, s)]
    out, new = ffn_width_residual(z, s)
    expected = a + b / (b.square().mean(-1, keepdim=True)+1e-6).sqrt() * a.square().mean(-1, keepdim=True).sqrt()
    weight = torch.randn_like(out)
    grads = torch.autograd.grad((out*weight + .17*new.square()).sum(), (z,s))
    refs = torch.autograd.grad((expected*weight + .17*(a+b).square()).sum(), (a,b))
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)
    for got, want in zip(grads, refs):
        torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-11)


def test_gradcheck():
    args = tuple(torch.randn(1,2,4,device='cuda',dtype=torch.float64,requires_grad=True) for _ in range(2))
    assert torch.autograd.gradcheck(ffn_width_residual, args)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_old_state_original_z_and_per_token_scale(dtype):
    z = torch.rand(2,3,32,device='cuda',dtype=dtype)
    s = torch.zeros_like(z)
    out, first = ffn_width_residual(z,s)
    assert torch.equal(out,z) and torch.equal(first,z) and not s.any()
    later = z*2
    mixed, second = ffn_width_residual(later,first)
    assert torch.equal(second,first+later) and torch.equal(first,z)
    assert not torch.equal(second,first+mixed)
    r = (mixed.float()-later.float()).square().mean(-1).sqrt()
    target = later.float().square().mean(-1).sqrt()
    torch.testing.assert_close(r,target,rtol=.015,atol=1e-5)
    changed = first.clone();changed[0,0].zero_()
    other, _ = ffn_width_residual(later,changed)
    assert torch.equal(other[1],mixed[1]) and torch.equal(other[0,1:],mixed[0,1:])


def test_depth_gradient_not_detached():
    zs = [torch.rand(1,2,16,device='cuda',requires_grad=True) for _ in range(3)]
    state = torch.zeros_like(zs[0])
    for z in zs:
        out, state = ffn_width_residual(z,state)
    weights = torch.randn_like(out)
    grads = torch.autograd.grad((out*weights).sum(),zs)
    assert all(g.norm()>0 and torch.isfinite(g).all() for g in grads)
    torch.testing.assert_close(state,zs[0]+zs[1]+zs[2],rtol=0,atol=0)


@pytest.mark.parametrize('compiled', [False,True])
@pytest.mark.parametrize('zero_z,zero_s', [(True,True),(True,False),(False,True)])
def test_zero_norm_backward_is_finite(compiled,zero_z,zero_s):
    z,s = [(torch.zeros if zero else torch.randn)(1,2,32,device='cuda',dtype=torch.bfloat16).requires_grad_()
           for zero in (zero_z,zero_s)]
    fn = torch.compile(ffn_width_residual,fullgraph=True) if compiled else ffn_width_residual
    out,new = fn(z,s)
    grads = torch.autograd.grad(out.float().sum()+new.float().sum(),(z,s))
    assert all(torch.isfinite(x).all() for x in (out,new,*grads))


def test_first_block_and_complete_manual_unroll():
    model = build()
    idx = torch.randint(0,256,(2,128),device='cuda')
    x = norm(model.wte(idx));cos,sin = model.cos,model.sin
    state = x.new_zeros(2,128,512)
    normal = model.blocks[0](x,cos,sin)
    first,_ = model.blocks[0].forward_ffn_width(x,state,cos,sin)
    assert torch.equal(first,normal)
    for b in model.blocks:
        x = x+b.attention_branch(x,cos,sin)
        z = b.mlp.c_fc(norm(x)).relu().square()
        sf,zf = state.float(),z.float()
        r = sf/(sf.square().mean(-1,keepdim=True)+1e-6).sqrt() * zf.square().mean(-1,keepdim=True).sqrt()
        x = x+b.mlp.c_proj(z+r.to(z.dtype))
        state = state+z
    expected = norm(x)
    actual = model.hidden(idx)
    assert relative(actual,expected)<.003
    cotangent = torch.randn_like(actual)
    params = list(model.parameters())[:-1]  # head is not used by hidden()
    got = torch.autograd.grad((actual*cotangent).sum(),params)
    want = torch.autograd.grad((expected*cotangent).sum(),params)
    assert relative(torch.cat([g.flatten() for g in got]),torch.cat([g.flatten() for g in want]))<.015


def test_no_parameters_rng_optimizer_or_initial_logits_change():
    base=build(False,nonzero=False);rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone()
    new=build(True,nonzero=False)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state())
    equal_tree(base.state_dict(),new.state_dict())
    assert set(dict(new.named_buffers()))=={'cos','sin'}
    opts=[optimizer_for(m) for m in (base,new)]
    assert opts[0][1]==opts[1][1]
    equal_tree(opts[0][0].state_dict(),opts[1][0].state_dict())
    x=torch.randint(0,256,(2,128),device='cuda')
    assert torch.equal(base(x),new(x))


def test_forward_reset_and_batch_independence():
    model=build().eval()
    x=torch.randint(0,256,(2,128),device='cuda')
    with torch.no_grad():
        first=model(x);model(x.flip(1));again=model(x)
        alone=model(x[:1])
    assert torch.equal(first,again)
    assert relative(first[:1],alone)<.003


@pytest.mark.parametrize('kernel', ['original','fla_triton'])
def test_real_2k_causality(kernel):
    model=build(sequence_length=2048,attention_kernel=kernel)
    x=torch.randint(0,256,(1,2048),device='cuda');changed=x.clone()
    changed[:,1024:]=(changed[:,1024:]+1)%256
    with torch.no_grad():
        a=model(x);b=model(changed)
    assert torch.equal(a[:,:1024],b[:,:1024])
    assert not torch.equal(a[:,1024:],b[:,1024:])


@pytest.mark.parametrize('kernel', ['original','fla_triton'])
def test_compile_checkpoint_and_cce(kernel):
    models=[build(attention_kernel=kernel),build(compiled=True,attention_kernel=kernel),
            build(compiled=True,attention_kernel=kernel,activation_checkpoint=True)]
    x=torch.randint(0,256,(2,128),device='cuda');y=(x+1)%256
    values=[];grads=[]
    for m in models:
        loss=m(x,y);loss.backward();values.append(float(loss))
        grads.append(torch.cat([p.grad.float().flatten() for p in m.parameters()]))
    for loss,grad in zip(values[1:],grads[1:]):
        assert abs(loss-values[0])<.002
        assert relative(grad,grads[0])<.025
    assert abs(float(models[0](x,y,dense=True))-values[0])<.002


def test_cached_decode():
    from nanochat.engine import KVCache
    model=build().eval();x=torch.randint(0,256,(1,128),device='cuda')
    cache=KVCache(1,1,128,128,3,'cuda',torch.bfloat16)
    with torch.no_grad():
        expected=model(x)
        pieces=[model(x[:,:120],kv_cache=cache)]
        for t in range(120,128):pieces.append(model(x[:,t:t+1],kv_cache=cache))
    actual=torch.cat(pieces,dim=1)
    assert cache.get_pos()==128
    assert relative(actual,expected)<.01


def test_checkpoint_optimizer_rng_and_config(tmp_path):
    model=build();opt,wd=optimizer_for(model)
    x=torch.randint(0,256,(2,128),device='cuda');y=(x+1)%256
    for step in range(3):
        schedule(opt,step,305176,wd);opt.zero_grad(set_to_none=True)
        model(x,y).backward();opt.step()
    gradients={name:p.grad.clone() for name,p in model.named_parameters()}
    state=snapshot(model,opt,3,config=asdict(model.config),next_data_row=48)
    path=tmp_path/'latest.pt';torch.save(state,path)
    loaded=torch.load(path,map_location='cpu',weights_only=False)
    restored=ConnectionModel(ConnectionConfig.from_checkpoint(loaded['config'])).prepare()
    ropt,rwd=optimizer_for(restored)
    restored.load_state_dict(loaded['model']);ropt.load_state_dict(loaded['optimizer'])
    equal_tree(model.state_dict(),restored.state_dict());equal_tree(opt.state_dict(),ropt.state_dict())
    restore_rng(state['rng']);expected=(torch.rand(8),torch.rand(8,device='cuda'))
    restore_rng(loaded['rng']);equal_tree(expected,(torch.rand(8),torch.rand(8,device='cuda')))
    for m,o,w in ((model,opt,wd),(restored,ropt,rwd)):
        for name,p in m.named_parameters():p.grad=gradients[name].clone()
        schedule(o,3,305176,w);o.step()
    equal_tree(model.state_dict(),restored.state_dict());equal_tree(opt.state_dict(),ropt.state_dict())
    assert loaded['next_data_row']==48 and restored.config.ffn_width_residual
    old=asdict(model.config);old.pop('ffn_width_residual')
    assert not ConnectionConfig.from_checkpoint(old).ffn_width_residual


def test_cost_and_unsupported_configs():
    from nanochat.connection_cost import model_cost
    base=model_cost(ConnectionConfig(),batch=16)
    new=model_cost(ConnectionConfig(ffn_width_residual=True),batch=16)
    assert base['estimated_training_flops']==new['estimated_training_flops']
    assert new['ffn_width_residual']['live_state_bytes']==128*1024**2
    for args in ({'architecture':'native'},{'variant':'mhc'},{'variant':'mhc_no_sinkhorn'}):
        with pytest.raises(ValueError,match='ordinary vanilla'):
            ConnectionConfig(ffn_width_residual=True,**args)
