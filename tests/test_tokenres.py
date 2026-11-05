"""TokenRes equations, shared gradients, causal support and training state."""
import copy
from dataclasses import asdict,replace
import pytest
import torch
from torch.nn import functional as F
from nanochat import flash_attention
from nanochat.tokenres import TokenResidual
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for,schedule
from nanochat.connection_experiment import configure_attention
from nanochat.connection_cost import model_cost
from nanochat.gpt import norm
from nanochat.engine import KVCache
from nanochat.packed_training import seed_all,snapshot,restore_rng,train_step
from test_connection_training_resume import equal_tree
from test_fla_attention import relative,record


@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(2);seed_all(42);torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.recompile_limit=64;configure_attention('fa3')


def build(**opts):
    args=dict(depth=2,width=128,heads=1,sequence_length=129,vocab_size=256,tokenres=True)
    args.update(opts)
    return ConnectionModel(ConnectionConfig(**args)).prepare()


def dense(q,k,v,window):
    q,k,v=(a.transpose(1,2) for a in (q,k,v));t=q.shape[-2]
    pos=torch.arange(t,device=q.device);distance=pos[:,None]-pos[None,:]
    mask=(distance>=0)&(distance<window)
    dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
    a=((q.to(dtype)@k.to(dtype).transpose(-1,-2))*q.shape[-1]**-.5).masked_fill(~mask,-torch.inf).softmax(-1)
    return (a@v.to(dtype)).to(v.dtype).transpose(1,2)


def oracle(module,x,u):
    b,t,d=x.shape;shape=(b,t,module.heads,module.head_dim)
    un=F.rms_norm(u,(d,));xn=F.rms_norm(x,(d,))
    q=F.linear(un,module.q.weight.to(u.dtype)).view(shape)
    k=F.linear(xn,module.k.weight.to(x.dtype)).view(shape)
    v=F.linear(xn,module.v.weight.to(x.dtype)).view(shape)
    q=F.rms_norm(q,(module.head_dim,))*1.2
    k=F.rms_norm(k,(module.head_dim,))*1.2
    return u+F.linear(dense(q,k,v,module.window).reshape(b,t,d),module.o.weight.to(x.dtype))


def test_fullsize_initialization_and_parameter_sharing():
    seed_all(42);base=ConnectionModel(ConnectionConfig());rng=torch.get_rng_state()
    seed_all(42);model=ConnectionModel(ConnectionConfig(tokenres=True));assert torch.equal(rng,torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters())==67108864
    assert len({id(b.token_residual) for b in model.blocks})==8
    old=base.state_dict()
    for name,p in model.state_dict().items():
        if 'token_residual' in name:continue
        if name.endswith(('attn.c_proj.weight','mlp.c_proj.weight')):assert p.norm()>0
        else:assert torch.equal(p,old[name]),name
    for block in model.blocks:
        assert len(list(block.token_residual.parameters()))==4
        assert all(p.shape==(512,512) and p.norm()>0 for p in block.token_residual.parameters())
        assert not torch.equal(block.token_residual.q.weight,block.token_residual.k.weight)
    assert not torch.equal(model.blocks[0].token_residual.q.weight,model.blocks[1].token_residual.q.weight)
    legacy=asdict(base.config);legacy.pop('tokenres');legacy.pop('tokenres_window')
    assert not ConnectionConfig.from_checkpoint(legacy).tokenres


def test_small_nonzero_output_initialization_without_runtime_scale():
    module=TokenResidual(512,4)
    # Only O uses a smaller initial distribution. All four matrices are learned
    # normally; there is no alpha parameter/buffer or zero-gradient barrier.
    assert module.o.weight.abs().max()<=.1*(3/512)**.5
    assert .095<float(module.o.weight.std()/module.v.weight.std())<.105
    assert not list(module.buffers())
    assert set(dict(module.named_parameters()))=={'q.weight','k.weight','v.weight','o.weight'}


@pytest.mark.parametrize('window',[1,7,128])
def test_fp64_equation_and_gradients(window,monkeypatch):
    # FA3 is BF16-only; substitute only its primitive with independent FP64
    # attention to audit the exact projection/addition graph separately.
    def primitive(q,k,v,*,causal,window_size):
        assert causal and window_size==(window-1,0)
        return dense(q,k,v,window)
    monkeypatch.setattr(flash_attention,'flash_attn_func',primitive)
    module=TokenResidual(32,2,window).cuda().double();ref=copy.deepcopy(module)
    xs=[torch.randn(2,17,32,device='cuda',dtype=torch.float64,requires_grad=True) for _ in range(2)]
    rs=[x.detach().clone().requires_grad_() for x in xs];g=torch.randn_like(xs[0])
    a=module(*xs);b=oracle(ref,*rs)
    ag=torch.autograd.grad((a*g).sum(),[*xs,*module.parameters()]);bg=torch.autograd.grad((b*g).sum(),[*rs,*ref.parameters()])
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    for x,y in zip(ag,bg):torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('t,window',[(17,1),(129,7),(257,128),(2048,128)])
def test_fa3_equation_and_vjp(t,window):
    m=TokenResidual(128,1,window).cuda();ref=copy.deepcopy(m)
    xs=[torch.randn(1,t,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(2)]
    rs=[x.detach().clone().requires_grad_() for x in xs]
    a=m(*xs);b=oracle(ref,*rs);g=torch.randn_like(a)
    ag=torch.autograd.grad((a.float()*g.float()).sum(),[*xs,*m.parameters()]);bg=torch.autograd.grad((b.float()*g.float()).sum(),[*rs,*ref.parameters()])
    errors=[relative(x,y) if y.norm()>1e-5 else float(x.abs().max()) for x,y in zip(ag,bg)]
    result=dict(forward=relative(a,b),gradients=errors);record(f'tokenres-{t}-{window}',result)
    assert result['forward']<.025 and max(errors)<.035,result


def test_exact_support_includes_diagonal_and_no_identity_bypass():
    m=TokenResidual(128,1,128).cuda()
    with torch.no_grad():
        m.q.weight.zero_();m.k.weight.zero_();m.v.weight.copy_(torch.eye(128,device='cuda'));m.o.weight.copy_(torch.eye(128,device='cuda'))
    x=torch.randn(1,257,128,device='cuda',dtype=torch.bfloat16,requires_grad=True);u=torch.zeros_like(x,requires_grad=True)
    y=m(x,u);g,=torch.autograd.grad(y[:,200].float().sum(),x)
    assert torch.equal((g[0,:,0]!=0).nonzero().flatten(),torch.arange(73,201,device='cuda'))
    torch.testing.assert_close(y[:,0],norm(x)[:,0],rtol=0,atol=0)
    with torch.no_grad():m.o.weight.zero_()
    u=torch.randn_like(x);torch.testing.assert_close(m(x,u),u,rtol=0,atol=0)


def test_block_formula_and_one_shared_module_called_twice():
    model=build();block=model.blocks[0];ref=copy.deepcopy(block)
    x=torch.randn(2,129,128,device='cuda',dtype=torch.bfloat16,requires_grad=True);rx=x.detach().clone().requires_grad_()
    calls=[];hook=block.token_residual.register_forward_hook(lambda m,ins,out:calls.append((ins[0],ins[1],out)))
    actual=block(x,model.cos,model.sin);hook.remove()
    u=ref.attention_branch(rx,model.cos,model.sin);middle=oracle(ref.token_residual,rx,u)
    expected=oracle(ref.token_residual,middle,ref.mlp_branch(middle))
    assert len(calls)==2 and calls[0][0] is x and calls[1][0] is calls[0][2]
    g=torch.randn_like(actual);(actual.float()*g.float()).sum().backward();(expected.float()*g.float()).sum().backward()
    result=dict(output=relative(actual,expected),gradient=relative(torch.cat([p.grad.flatten() for p in block.parameters()]),torch.cat([p.grad.flatten() for p in ref.parameters()])))
    record('tokenres-block',result);assert result['output']<.025 and result['gradient']<.035,result


def test_shared_parameter_gradient_is_sum_of_two_uses():
    model=build();block=model.blocks[0];ref=copy.deepcopy(block);second=copy.deepcopy(ref.token_residual)
    x=torch.randn(2,129,128,device='cuda',dtype=torch.bfloat16);a=block(x,model.cos,model.sin)
    u=ref.attention_branch(x,model.cos,model.sin);h=ref.token_residual(x,u);b=second(h,ref.mlp_branch(h));g=torch.randn_like(a)
    (a.float()*g.float()).sum().backward();(b.float()*g.float()).sum().backward()
    for p,p1,p2 in zip(block.token_residual.parameters(),ref.token_residual.parameters(),second.parameters()):
        assert relative(p.grad,p1.grad+p2.grad)<.02


def test_real_2k_causality():
    model=build(sequence_length=2048);x=torch.randint(0,256,(1,2048),device='cuda');other=x.clone();other[:,1024:]=(other[:,1024:]+7)%256
    with torch.no_grad():torch.testing.assert_close(model(x)[:,:1024],model(other)[:,:1024],rtol=0,atol=0)
    module=model.blocks[0].token_residual
    xs=[torch.randn(1,2048,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(2)]
    g=torch.autograd.grad(module(*xs)[:,:1024].float().square().sum(),xs)
    assert all(a[:,1024:].count_nonzero()==0 for a in g)


def test_compile_checkpoint_and_dense_ce():
    eager=build();idx=torch.randint(0,256,(2,129),device='cuda');target=(idx+1)%256;losses=[];gradients=[]
    for kind in ('eager','compile','checkpoint','dense'):
        m=eager if kind=='eager' else ConnectionModel(replace(eager.config,activation_checkpoint=kind=='checkpoint')).prepare(compile_blocks=kind in ('compile','checkpoint'))
        if m is not eager:m.load_state_dict(eager.state_dict())
        loss=m(idx,target,dense=kind=='dense');loss.backward();losses.append(float(loss));gradients.append(torch.cat([p.grad.float().flatten() for p in m.parameters()]))
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    result=dict(losses=losses,gradient_relative=[relative(g,gradients[0]) for g in gradients[1:]]);record('tokenres-compile',result)
    assert max(losses)-min(losses)<.002 and max(result['gradient_relative'])<.02,result


def test_exact_optimizer_rng_resume(tmp_path):
    model=build();opt,wd=optimizer_for(model);x=torch.randint(0,256,(2,129),device='cuda');y=(x+1)%256
    for step in range(3):
        schedule(opt,step,305176,wd);opt.zero_grad(set_to_none=True);model(x,y).backward();opt.step()
    grads={n:p.grad.clone() for n,p in model.named_parameters()};path=tmp_path/'snapshot.pt'
    torch.save(snapshot(model,opt,3,config=asdict(model.config)),path);state=torch.load(path,map_location='cpu',weights_only=False)
    other=ConnectionModel(ConnectionConfig.from_checkpoint(state['config'])).prepare();oo,_=optimizer_for(other)
    other.load_state_dict(state['model']);oo.load_state_dict(state['optimizer'])
    equal_tree(model.state_dict(),other.state_dict());equal_tree(opt.state_dict(),oo.state_dict())
    restore_rng(state['rng']);expected=(torch.rand(8),torch.rand(8,device='cuda'))
    restore_rng(state['rng']);equal_tree(expected,(torch.rand(8),torch.rand(8,device='cuda')))
    for m,o in ((model,opt),(other,oo)):
        for n,p in m.named_parameters():p.grad=grads[n].clone()
        schedule(o,3,305176,wd);o.step()
    equal_tree(model.state_dict(),other.state_dict());equal_tree(opt.state_dict(),oo.state_dict())


def test_optimizer_covers_shared_weights_once_and_preserves_lr():
    m=build();opt,wd=optimizer_for(m);base=build(tokenres=False);bo,bwd=optimizer_for(base);assert wd==bwd
    ids=[id(p) for group in opt.param_groups for p in group['params']];assert len(ids)==len(set(ids))==len(list(m.parameters()))
    for block in m.blocks:
        for p in block.token_residual.parameters():
            group=next(g for g in opt.param_groups if any(p is q for q in g['params']))
            assert group['kind']=='muon' and group['lr']==.005


@pytest.mark.parametrize('opts',[dict(architecture='native'),dict(variant='mhc'),dict(attention_kernel='fla_triton'),dict(tokenres_window=0),dict(tokenres_window=1.5)])
def test_invalid_combinations(opts):
    with pytest.raises(ValueError):ConnectionConfig(tokenres=True,**opts)


def test_cached_decoding_rejected_before_mutation():
    m=build();cache=KVCache(1,1,129,128,2,'cuda',torch.bfloat16)
    with pytest.raises(ValueError,match='separate sublayer KV'):m(torch.zeros(1,1,device='cuda',dtype=torch.long),kv_cache=cache)
    assert cache.get_pos()==0


def test_no_slow_attention_fallback(monkeypatch):
    monkeypatch.setattr(flash_attention,'USE_FA3',False)
    m=TokenResidual(128,1).cuda();x=torch.zeros(1,17,128,device='cuda',dtype=torch.bfloat16)
    with pytest.raises(RuntimeError,match='no dense fallback'):m(x,x)


def test_flops_account_for_both_tokenres_calls():
    c=ConnectionConfig(tokenres=True);base=model_cost(replace(c,tokenres=False));new=model_cost(c)
    t,d,l,w=c.sequence_length,c.width,c.depth,c.tokenres_window
    extra=16*t*l*d*d+8*d*l*(w*t-w*(w-1)//2)
    assert new['estimated_forward_flops']==base['estimated_forward_flops']+extra
    assert new['estimated_training_flops']==3*new['estimated_forward_flops']


def test_pre_and_qk_norm_match_backbone(monkeypatch):
    module=TokenResidual(32,2).cuda().double()
    x=torch.randn(2,17,32,device='cuda',dtype=torch.float64)*1000
    u=torch.randn_like(x)*100
    def rms(z):
        return z/(z.square().mean(-1,keepdim=True)+torch.finfo(z.dtype).eps).sqrt()
    expected=[rms(F.linear(rms(u),module.q.weight).view(2,17,2,16))*1.2,
              rms(F.linear(rms(x),module.k.weight).view(2,17,2,16))*1.2,
              F.linear(rms(x),module.v.weight).view(2,17,2,16)]
    def primitive(q,k,v,**kwargs):
        for a,b in zip((q,k,v),expected):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        return dense(q,k,v,128)
    monkeypatch.setattr(flash_attention,'flash_attn_func',primitive)
    module(x,u)


def test_raw_checkpoint_not_silently_reinterpreted():
    config=asdict(ConnectionConfig(tokenres=True));config.pop('tokenres_normalization')
    with pytest.raises(ValueError,match='archived source'):ConnectionConfig.from_checkpoint(config)
    config['tokenres']=False
    assert not ConnectionConfig.from_checkpoint(config).tokenres


def test_nonfinite_post_optimizer_is_rejected(monkeypatch):
    model=build(sequence_length=128);opt,_=optimizer_for(model)
    x=torch.randint(0,256,(1,128),device='cuda');y=(x+1)%256
    class Data:
        def batch(self,*args):return x,y
    @torch.no_grad()
    def corrupt():model.blocks[0].token_residual.o.weight.fill_(float('nan'))
    monkeypatch.setattr(opt,'step',corrupt)
    with pytest.raises(FloatingPointError,match='after optimizer'):
        train_step(model,opt,Data(),0,microbatch=1,batch_tokens=128)
