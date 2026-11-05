"""Dense wide-state routing, unchanged branches, gradients and causal execution."""
from dataclasses import asdict
import gc
import pytest
import torch
import torch.nn.functional as F

from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.connection_experiment import configure_attention
from nanochat.gpt import norm
from nanochat.wide_residual import WideResidualConnection
from nanochat.packed_training import seed_all, snapshot, restore_rng
from test_connection_training_resume import equal_tree


@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(2);seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.recompile_limit=64
    configure_attention('fa3')
    yield
    gc.collect();torch.cuda.empty_cache()


def build(enabled=True, compiled=False, nonzero=True, **kwargs):
    seed_all(42)
    options=dict(depth=3,width=128,heads=1,vocab_size=256,sequence_length=128,wide_residual=enabled)
    options.update(kwargs)
    model=ConnectionModel(ConnectionConfig(**options)).prepare(compile_blocks=compiled)
    if nonzero:
        with torch.no_grad():
            for b in model.blocks:
                b.attn.c_proj.weight.normal_(std=.02)
                b.mlp.c_proj.weight.normal_(std=.02)
    return model


def relative(a,b):
    return float(((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-9)).detach())


def test_connection_formula_and_all_gradients_double():
    c=WideResidualConnection(8).cuda().double()
    s=torch.randn(2,3,32,device='cuda',dtype=torch.float64,requires_grad=True)
    params=[s,c.down_proj.weight,c.up_proj.weight]
    refs=[v.detach().clone().requires_grad_() for v in params]
    x=c.read(s);y=x.relu().square();got=c.write(s,y)
    a,down,up=refs
    normalized=a/(a.square().mean(-1,keepdim=True)+torch.finfo(a.dtype).eps).sqrt()
    expected=a+F.linear(F.linear(normalized,down).relu().square(),up)
    torch.testing.assert_close(got,expected,rtol=1e-12,atol=1e-12)
    weight=torch.randn_like(got)
    gs=torch.autograd.grad((got*weight).sum(),params)
    rs=torch.autograd.grad((expected*weight).sum(),refs)
    for g,r in zip(gs,rs):torch.testing.assert_close(g,r,rtol=1e-10,atol=1e-11)


def test_wide_norm_and_no_post_read_norm():
    c=WideResidualConnection(8).cuda().double()
    s=torch.randn(2,3,32,device='cuda',dtype=torch.float64)
    s[...,:8]*=20
    with torch.no_grad():c.down_proj.weight.mul_(3)
    read=c.read(s)
    torch.testing.assert_close(read,F.linear(norm(s),c.down_proj.weight),rtol=0,atol=0)
    assert not torch.allclose(read,norm(read))
    zero=torch.zeros_like(read)
    assert torch.equal(c.write(s,zero),s)
    assert set(dict(c.named_parameters()))=={'down_proj.weight','up_proj.weight'}


def test_projection_shapes_rank_independence_and_optimizer():
    m=build(nonzero=False)
    projections=[m.init_up_proj,m.final_down_proj]
    for b in m.blocks:
        for c in (b.attn_wide,b.mlp_wide):
            assert c.down_proj.weight.shape==(128,512)
            assert c.up_proj.weight.shape==(512,128)
            projections.extend((c.down_proj,c.up_proj))
    assert len({p.weight.data_ptr() for p in projections})==len(projections)==14
    for p in projections:
        assert p.bias is None
        assert torch.linalg.matrix_rank(p.weight.double()).item()==128
    opt,wd=optimizer_for(m)
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(m.parameters()))
    muon={id(p) for g in opt.param_groups if g['kind']=='muon' for p in g['params']}
    assert all(id(p.weight) in muon for p in projections)
    base=build(False,nonzero=False);bo,bwd=optimizer_for(base)
    assert wd==bwd
    assert [{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups]==[
            {k:v for k,v in g.items() if k!='params'} for g in bo.param_groups]


def test_original_parameters_and_rng_preserved():
    base=build(False,nonzero=False);rng=torch.get_rng_state();crng=torch.cuda.get_rng_state()
    wide=build(True,nonzero=False)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state())
    for name,value in base.state_dict().items():assert torch.equal(value,wide.state_dict()[name]),name
    for b in wide.blocks:
        assert b.mlp.c_fc.weight.shape==(512,128)  # original4D ReLU squared, no gate
        assert not b.attn.c_proj.weight.any() and not b.mlp.c_proj.weight.any()
    assert set(dict(wide.named_buffers()))=={'cos','sin'}


def test_identity_wide_path_forward_and_backward():
    m=build(nonzero=False)
    s=torch.randn(2,128,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=m.blocks[0](s,m.cos,m.sin)
    assert torch.equal(y,s)
    cotangent=torch.randn_like(y)
    gradient=torch.autograd.grad((y*cotangent).sum(),s)[0]
    assert torch.equal(gradient,cotangent)


def test_complete_manual_unroll_and_all_gradients():
    m=build();idx=torch.randint(0,256,(2,128),device='cuda')
    s=F.linear(m.wte(idx),m.init_up_proj.weight.to(torch.bfloat16))
    for b in m.blocks:
        x=F.linear(norm(s),b.attn_wide.down_proj.weight.to(s.dtype))
        y=b.attn(x,None,(m.cos,m.sin),(-1,0),None)
        s=s+F.linear(y,b.attn_wide.up_proj.weight.to(s.dtype))
        x=F.linear(norm(s),b.mlp_wide.down_proj.weight.to(s.dtype))
        z=F.relu(b.mlp.c_fc(x)).square()
        s=s+F.linear(b.mlp.c_proj(z),b.mlp_wide.up_proj.weight.to(s.dtype))
    h=norm(F.linear(s,m.final_down_proj.weight.to(s.dtype)))
    expected=15*torch.tanh(F.linear(h,m.lm_head.weight.to(h.dtype)).float()/15)
    actual=m(idx)
    assert torch.equal(actual,expected)
    cotangent=torch.randn_like(actual)
    got=torch.autograd.grad((actual*cotangent).sum(),list(m.parameters()))
    want=torch.autograd.grad((expected*cotangent).sum(),list(m.parameters()))
    assert relative(torch.cat([x.flatten() for x in got]),torch.cat([x.flatten() for x in want]))<.01
    assert all(torch.isfinite(x).all() and x.norm()>0 for x in got)


def test_initial_and_final_projection_and_reset():
    m=build(nonzero=False).eval();x=torch.randint(0,256,(2,128),device='cuda')
    with torch.no_grad():
        expected=norm(m.final_down_proj(m.init_up_proj(m.wte(x))))
        actual=m.hidden(x);m.hidden(x.flip(1));again=m.hidden(x)
    assert torch.equal(actual,expected) and torch.equal(actual,again)
    assert torch.isfinite(m(x)).all()


@pytest.mark.parametrize('kernel',['original','fla_triton'])
def test_real_2k_causality_and_batch_independence(kernel):
    m=build(sequence_length=2048,attention_kernel=kernel)
    x=torch.randint(0,256,(2,2048),device='cuda');changed=x.clone()
    changed[0,1024:]=(changed[0,1024:]+1)%256
    with torch.no_grad():a=m(x);b=m(changed)
    assert torch.equal(a[0,:1024],b[0,:1024]) and torch.equal(a[1],b[1])
    assert not torch.equal(a[0,1024:],b[0,1024:])


@pytest.mark.parametrize('kernel',['original','fla_triton'])
def test_compile_checkpoint_and_cce(kernel):
    models=[build(attention_kernel=kernel),build(compiled=True,attention_kernel=kernel),
            build(compiled=True,attention_kernel=kernel,activation_checkpoint=True)]
    x=torch.randint(0,256,(2,128),device='cuda');y=(x+1)%256
    losses=[];grads=[]
    for m in models:
        loss=m(x,y);loss.backward();losses.append(float(loss.detach()))
        grads.append(torch.cat([p.grad.float().flatten() for p in m.parameters()]))
    for loss,grad in zip(losses[1:],grads[1:]):
        assert abs(loss-losses[0])<.002
        assert relative(grad,grads[0])<.025
    assert abs(float(models[0](x,y,dense=True).detach())-losses[0])<.002


def test_cached_decode():
    from nanochat.engine import KVCache
    m=build().eval();x=torch.randint(0,256,(1,128),device='cuda')
    cache=KVCache(1,1,128,128,3,'cuda',torch.bfloat16)
    with torch.no_grad():
        expected=m(x);pieces=[m(x[:,:120],kv_cache=cache)]
        for t in range(120,128):pieces.append(m(x[:,t:t+1],kv_cache=cache))
    assert cache.get_pos()==128
    assert relative(torch.cat(pieces,dim=1),expected)<.01


def test_checkpoint_optimizer_rng_and_fixed_gradient_next_update(tmp_path):
    m=build();opt,wd=optimizer_for(m)
    x=torch.randint(0,256,(2,128),device='cuda');y=(x+1)%256
    for step in range(3):
        schedule(opt,step,305176,wd);opt.zero_grad(set_to_none=True);m(x,y).backward();opt.step()
    gradients={n:p.grad.clone() for n,p in m.named_parameters()}
    state=snapshot(m,opt,3,config=asdict(m.config),next_data_row=48)
    path=tmp_path/'latest.pt';torch.save(state,path)
    loaded=torch.load(path,map_location='cpu',weights_only=False)
    other=ConnectionModel(ConnectionConfig.from_checkpoint(loaded['config'])).prepare()
    oo,ow=optimizer_for(other);other.load_state_dict(loaded['model']);oo.load_state_dict(loaded['optimizer'])
    equal_tree(m.state_dict(),other.state_dict());equal_tree(opt.state_dict(),oo.state_dict())
    restore_rng(state['rng']);expected=(torch.rand(8),torch.rand(8,device='cuda'))
    restore_rng(loaded['rng']);equal_tree(expected,(torch.rand(8),torch.rand(8,device='cuda')))
    for model,o,w in ((m,opt,wd),(other,oo,ow)):
        for n,p in model.named_parameters():p.grad=gradients[n].clone()
        schedule(o,3,305176,w);o.step()
    equal_tree(m.state_dict(),other.state_dict());equal_tree(opt.state_dict(),oo.state_dict())
    old=asdict(m.config);old.pop('wide_residual')
    assert not ConnectionConfig.from_checkpoint(old).wide_residual
    with pytest.raises(ValueError,match='retired FFN-width'):
        ConnectionConfig.from_checkpoint(dict(old,ffn_width_residual=True))


def test_full_size_parameters_cost_and_unsupported_configs():
    from nanochat.connection_cost import model_cost
    m=build(depth=8,width=512,heads=4,vocab_size=32768,sequence_length=2048,nonzero=False)
    assert sum(p.numel() for p in m.parameters())==94371840
    c=m.config;wide=model_cost(c,batch=16);base=model_cost(ConnectionConfig(),batch=16)
    extra=2*16*2048*34*512*2048
    assert wide['estimated_forward_flops']-base['estimated_forward_flops']==extra
    for args in ({'architecture':'native'},{'variant':'mhc'},{'variant':'mhc_no_sinkhorn'}):
        with pytest.raises(ValueError,match='ordinary vanilla'):
            ConnectionConfig(wide_residual=True,**args)
