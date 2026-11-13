"""GPU-only checks for the dense vocabulary gate and its fused CE."""
from dataclasses import asdict
import pytest
import torch
from torch.nn import functional as F
from nanochat.vocab_gate import gate_logits, gate_loss
from nanochat.vocab_gate_triton import linear_gate_cross_entropy
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for,schedule
from nanochat.connection_experiment import configure_attention
from nanochat.packed_training import seed_all,snapshot,restore_rng


def close(a,b,tol=.008):
    a,b=a.float(),b.float()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    rel=(a-b).norm()/b.norm().clamp_min(1e-12)
    peak=(a-b).abs().max()/b.abs().max().clamp_min(1e-12)
    assert rel<=tol and peak<=4*tol,(rel.item(),peak.item())


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('shape',[(1,64,97),(37,128,1024),(129,512,32768)])
@pytest.mark.parametrize('gate_scale',[.01,1.,20.])
@pytest.mark.parametrize('kernel',['row','tiled'])
def test_loss_and_all_gradients(dtype,shape,gate_scale,kernel):
    seed_all(42);torch.backends.cuda.matmul.allow_tf32=False
    n,d,v=shape
    h=torch.randn(n,d,device='cuda',dtype=dtype,requires_grad=True)
    w=(torch.randn(2*v,d,device='cuda')*.05).requires_grad_()
    with torch.no_grad(): w[v:].mul_(gate_scale)
    y=torch.randint(v,(n,),device='cuda')
    expected=gate_loss(h,w,y,backend='reference')
    eg=torch.autograd.grad(expected,(h,w))
    loss=linear_gate_cross_entropy(h,w,y,chunk_size=31,kernel=kernel)
    got=torch.autograd.grad(loss,(h,w))
    torch.testing.assert_close(loss,expected,rtol=2e-6,atol=3e-6)
    for a,b in zip(got,eg):close(a,b,.008 if dtype==torch.bfloat16 else 1e-5)


def test_independent_fp64_formula():
    z=torch.randn(4,7,device='cuda',dtype=torch.float64,requires_grad=True)
    g=torch.randn_like(z,requires_grad=True);y=torch.tensor([0,3,4,6],device='cuda')
    def f(z,g):return F.cross_entropy(15*torch.tanh(z*(1+.5*torch.tanh(g))/15),y)
    assert torch.autograd.gradcheck(f,(z,g),eps=1e-6,atol=1e-7,rtol=1e-5)
    logits=15*torch.tanh(z*(1+.5*torch.tanh(g))/15)
    e=logits.softmax(-1)-F.one_hot(y,7)
    dc=e/4*(1-torch.tanh(z*(1+.5*torch.tanh(g))/15).square())
    got=torch.autograd.grad(f(z,g),(z,g))
    torch.testing.assert_close(got[0],dc*(1+.5*torch.tanh(g)))
    torch.testing.assert_close(got[1],dc*z*.5*(1-torch.tanh(g).square()))


@pytest.mark.parametrize('grad_h,grad_w',[(True,False),(False,True),(True,True)])
def test_upstream_scaling_and_frozen_inputs(grad_h,grad_w):
    h=torch.randn(19,64,device='cuda',dtype=torch.bfloat16,requires_grad=grad_h)
    w=(torch.randn(514,64,device='cuda')*.02).requires_grad_(grad_w)
    y=torch.randint(257,(19,),device='cuda')
    params=[x for x in (h,w) if x.requires_grad]
    a=torch.autograd.grad(-.375*linear_gate_cross_entropy(h,w,y,chunk_size=7),params)
    b=torch.autograd.grad(-.375*gate_loss(h,w,y,backend='reference'),params)
    for x,z in zip(a,b):close(x,z)
    with torch.no_grad():
        torch.testing.assert_close(linear_gate_cross_entropy(h,w,y,chunk_size=7),gate_loss(h,w,y,backend='reference'),rtol=2e-6,atol=3e-6)


def build(gate=True,backend='triton',compiled=False,checkpoint=False,production=False):
    seed_all(42)
    c=ConnectionConfig(streams=1,vocab_gate=gate,vocab_gate_backend=backend,
        depth=8 if production else 2,width=512 if production else 128,
        heads=4 if production else 1,sequence_length=128 if production else 64,
        vocab_size=32768 if production else 512,activation_checkpoint=checkpoint)
    return ConnectionModel(c).prepare(compile_blocks=compiled)


def test_initialization_full_dense_gate_and_optimizer():
    configure_attention('fa2')
    base=build(False);rng=torch.get_rng_state().clone();m=build()
    assert torch.equal(rng,torch.get_rng_state())
    for name,p in base.named_parameters():
        q=dict(m.named_parameters())[name]
        if name=='lm_head.weight':q=q[:512]
        assert torch.equal(p,q),name
    assert m.lm_head.weight.shape==(1024,128)
    assert m.lm_head.weight[512:].count_nonzero()==0
    x=torch.randint(512,(2,64),device='cuda');y=(x+1)%512
    torch.testing.assert_close(m(x),base(x),rtol=0,atol=0)
    m(x,y).backward()
    assert m.lm_head.weight.grad[:512].norm()>0
    assert m.lm_head.weight.grad[512:].norm()>0
    opt,_=optimizer_for(m)
    assert len({id(p) for g in opt.param_groups for p in g['params']})==len(list(m.parameters()))


@pytest.mark.parametrize('compiled,checkpoint',[(False,False),(True,False),(True,True)])
def test_model_gradients(compiled,checkpoint):
    configure_attention('fa2');torch.backends.cuda.matmul.allow_tf32=False
    ref=build(backend='reference',compiled=compiled,checkpoint=checkpoint)
    actual=build(compiled=compiled,checkpoint=checkpoint)
    with torch.no_grad():
        ref.lm_head.weight[512:].normal_(0,.02)
        for b in ref.blocks:
            b.attn.c_proj.weight.normal_(0,.01);b.mlp.c_proj.weight.normal_(0,.01)
    actual.load_state_dict(ref.state_dict())
    x=torch.randint(512,(2,64),device='cuda');y=(x+1)%512
    a=ref(x,y);a.backward();b=actual(x,y);b.backward()
    torch.testing.assert_close(a,b,rtol=1e-5,atol=3e-6)
    for (name,p),(_,q) in zip(ref.named_parameters(),actual.named_parameters()):
        try:close(p.grad,q.grad,.02)
        except AssertionError as e:raise AssertionError(name,str(e)) from e


def test_production_width_all_parameter_gradients():
    configure_attention('fa2');torch.backends.cuda.matmul.allow_tf32=False
    ref=build(backend='reference',compiled=True,production=True)
    actual=build(compiled=True,production=True)
    with torch.no_grad():
        ref.lm_head.weight[32768:].normal_(0,.01)
        for b in ref.blocks:
            b.attn.c_proj.weight.normal_(0,.01);b.mlp.c_proj.weight.normal_(0,.01)
    actual.load_state_dict(ref.state_dict())
    x=torch.randint(32768,(2,128),device='cuda');y=(x+1)%32768
    a=ref(x,y);a.backward();b=actual(x,y);b.backward()
    torch.testing.assert_close(a,b,rtol=1e-5,atol=3e-6)
    for (name,p),(_,q) in zip(ref.named_parameters(),actual.named_parameters()):
        try:close(p.grad,q.grad,.02)
        except AssertionError as e:raise AssertionError(name,str(e)) from e


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal_tree(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    else:assert a==b


def test_checkpoint_real_kernel_next_update(tmp_path):
    configure_attention('fa2');m=build();opt,wd=optimizer_for(m)
    x=torch.randint(512,(2,64),device='cuda');y=(x+1)%512
    for step in range(3):
        opt.zero_grad(set_to_none=True);schedule(opt,step,100,wd);m(x,y).backward();opt.step()
    path=tmp_path/'state.pt';torch.save(snapshot(m,opt,3,config=asdict(m.config),next_data_row=6),path)
    s=torch.load(path,map_location='cpu',weights_only=False)
    other=ConnectionModel(ConnectionConfig.from_checkpoint(s['config'])).prepare()
    oo,ow=optimizer_for(other);other.load_state_dict(s['model']);oo.load_state_dict(s['optimizer'])
    equal_tree(m.state_dict(),other.state_dict());equal_tree(opt.state_dict(),oo.state_dict())
    restore_rng(s['rng']);r=(torch.rand(4),torch.rand(4,device='cuda'))
    restore_rng(s['rng']);equal_tree(r,(torch.rand(4),torch.rand(4,device='cuda')))
    assert s['step']==3 and s['next_data_row']==6
    for model,o,w in [(m,opt,wd),(other,oo,ow)]:
        o.zero_grad(set_to_none=True);schedule(o,3,100,w);model(x,y).backward();o.step()
    equal_tree(m.state_dict(),other.state_dict());equal_tree(opt.state_dict(),oo.state_dict())


def test_causal_cached_inference_nonzero_gate():
    from nanochat import flash_attention
    from nanochat.engine import KVCache
    flash_attention.USE_FA3=False;torch.backends.cuda.enable_math_sdp(True)
    m=build(compiled=True).eval()
    with torch.no_grad():
        m.lm_head.weight[512:].normal_(0,.02)
        for b in m.blocks:
            b.attn.c_proj.weight.normal_(0,.01);b.mlp.c_proj.weight.normal_(0,.01)
        x=torch.randint(512,(1,64),device='cuda');changed=x.clone();changed[:,31:]=(changed[:,31:]+7)%512
        full=m(x);close(m(changed)[:,:31],full[:,:31],.003)
        cache=KVCache(1,1,64,128,2,'cuda',torch.bfloat16)
        parts=[];start=0
        for size in (7,13,1,43):
            parts.append(m(x[:,start:start+size],kv_cache=cache));start+=size
        close(torch.cat(parts,1),full,.008);assert cache.get_pos()==64


def test_retired_and_invalid_config():
    for kw in [dict(vocab_gate=True,streams=4),dict(vocab_gate_alpha=2),dict(vocab_gate_chunk=0)]:
        with pytest.raises(ValueError):ConnectionConfig(**kw)
    with pytest.raises(ValueError,match='retired FFN'):
        ConnectionConfig.from_checkpoint({'ffn_readout_width':64})
    assert not ConnectionConfig.from_checkpoint({'ffn_readout_width':0,'ffn_readout_init':'gradmax'}).vocab_gate
