"""Shared attention refinement: FA3 local/full masks, VJP and training state."""
from dataclasses import asdict, replace
import pytest
import torch
from nanochat import flash_attention
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.connection_experiment import configure_attention
from nanochat.connection_cost import model_cost
from nanochat.gpt import norm, apply_rotary_emb
from nanochat.packed_training import seed_all, snapshot, restore_rng
from test_connection_training_resume import equal_tree
from test_triton_moments import relative, record

@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(2); seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.recompile_limit=64
    configure_attention('fa3')
    assert flash_attention.USE_FA3


def build(**options):
    args=dict(depth=2,width=128,heads=1,vocab_size=256,sequence_length=129,
              attention_looping=True)
    args.update(options)
    return ConnectionModel(ConnectionConfig(**args)).prepare()


def dense_attention(q,k,v,window):
    q,k,v=(x.transpose(1,2) for x in (q,k,v))
    t=q.shape[-2]; pos=torch.arange(t,device=q.device)
    gap=pos[:,None]-pos[None,:]
    mask=gap>=0
    if window is not None:mask=mask & (gap<window)
    probs=((q.float()@k.float().transpose(-1,-2))*q.shape[-1]**-.5).masked_fill(~mask,-torch.inf).softmax(-1)
    return (probs@v.float()).to(v.dtype).transpose(1,2)


@pytest.mark.parametrize('t,window',[(17,1),(129,7),(257,128),(2048,128),(257,512),(257,129)])
def test_fa3_local_forward_and_vjp(t,window):
    xs=[torch.randn(1,t,1,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    refs=[x.detach().clone().requires_grad_() for x in xs]
    actual=flash_attention.flash_attn_func(*xs,causal=True,window_size=(window-1,0))
    expected=dense_attention(*refs,window)
    upstream=torch.randn_like(actual)
    grads=torch.autograd.grad((actual.float()*upstream.float()).sum(),xs)
    refgrads=torch.autograd.grad((expected.float()*upstream.float()).sum(),refs)
    errors=[relative(a,b) if b.float().norm()>1e-5 else float(a.float().abs().max()) for a,b in zip(grads,refgrads)]
    result=dict(t=t,window=window,forward=relative(actual,expected),gradients=errors)
    record(f'fa3-window-{t}-{window}',result)
    assert result['forward']<.025 and max(errors)<.035,result
    assert all(torch.isfinite(x).all() for x in (actual,*grads))


def test_exact_window_boundary_and_future_exclusion():
    t=257;window=128;target=200
    q=k=torch.zeros(1,t,1,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(q).requires_grad_()
    out=flash_attention.flash_attn_func(q,k,v,causal=True,window_size=(window-1,0))
    grad,=torch.autograd.grad(out[:,target].float().sum(),v)
    visible=(grad[0,:,0,0]!=0).nonzero().flatten()
    assert torch.equal(visible,torch.arange(target-window+1,target+1,device='cuda'))
    changed=v.detach().clone();changed[:,:target-window+1]+=7;changed[:,target+1:]-=9
    new=flash_attention.flash_attn_func(q,k,changed,causal=True,window_size=(window-1,0))
    assert torch.equal(new[:,target],out[:,target])


def activate_branches(model):
    # Test trained-like nonzero outputs; zero-init would conceal ordering errors.
    with torch.no_grad():
        for block in model.blocks:
            block.attn.c_proj.weight.normal_(std=.02)
            block.mlp.c_proj.weight.normal_(std=.02)


def manual_hidden(model,idx):
    x=norm(model.wte(idx));b,t,d=x.shape
    for block in model.blocks:
        for i in range(model.config.attention_loops):
            h=norm(x);a=block.attn;shape=(b,t,a.n_head,a.head_dim)
            q=norm(apply_rotary_emb(a.c_q(h).view(shape),model.cos,model.sin))*1.2
            k=norm(apply_rotary_emb(a.c_k(h).view(shape),model.cos,model.sin))*1.2
            v=a.c_v(h).view(shape)
            window=model.config.attention_loop_window if i<model.config.attention_loops-1 else None
            x=x+a.c_proj(dense_attention(q,k,v,window).reshape(b,t,d))
        x=x+block.mlp(norm(x))
    return norm(x)


@pytest.mark.parametrize('loops',[2,3,4])
def test_shared_parameter_gradients_and_one_mlp(loops):
    model=build(attention_loops=loops);activate_branches(model)
    reference=build(attention_loops=loops);reference.load_state_dict(model.state_dict())
    x=torch.randint(0,256,(2,129),device='cuda');upstream=torch.randn(2,129,128,device='cuda')
    counts=dict(attention=0,mlp=0);windows=[];handles=[]
    def count_attention(module,args):counts['attention']+=1;windows.append(args[3])
    def count_mlp(module,args):counts['mlp']+=1
    for block in model.blocks:
        handles.extend([block.attn.register_forward_pre_hook(count_attention),block.mlp.register_forward_pre_hook(count_mlp)])
    actual=model.hidden(x);expected=manual_hidden(reference,x)
    for handle in handles:handle.remove()
    (actual.float()*upstream).sum().backward();(expected.float()*upstream).sum().backward()
    gradients=torch.cat([p.grad.float().flatten() for p in model.parameters() if p.grad is not None])
    refgradients=torch.cat([p.grad.float().flatten() for p in reference.parameters() if p.grad is not None])
    assert counts==dict(attention=2*loops,mlp=2)
    assert windows==([(127,0)]*(loops-1)+[(-1,0)])*2
    result=dict(loops=loops,hidden=relative(actual,expected),gradients=relative(gradients,refgradients))
    record(f'loop-model-{loops}',result)
    assert result['hidden']<.025 and result['gradients']<.035,result
    assert all(p.grad is not None and p.grad.norm()>0 for block in model.blocks for p in block.parameters())


def test_unchanged_parameters_rng_optimizer_and_single_loop():
    models=[];rngs=[]
    for settings in [dict(attention_looping=False),dict(attention_loops=1),{}]:
        seed_all(42);models.append(build(**settings));rngs.append(torch.get_rng_state().clone())
    base=models[0];state=base.state_dict()
    for model,rng in zip(models[1:],rngs[1:]):
        equal_tree(state,model.state_dict());assert torch.equal(rngs[0],rng)
        opt,_=optimizer_for(model)
        ids=[id(p) for group in opt.param_groups for p in group['params']]
        assert len(ids)==len(set(ids))==len(list(model.parameters()))
    activate_branches(base);models[1].load_state_dict(base.state_dict())
    x=torch.randint(0,256,(2,129),device='cuda')
    assert torch.equal(base(x),models[1](x))
    legacy=asdict(base.config)
    for key in ('attention_looping','attention_loops','attention_loop_window'):legacy.pop(key)
    assert not ConnectionConfig.from_checkpoint(legacy).attention_looping


def test_true_2k_causality():
    model=build(sequence_length=2048,depth=2);activate_branches(model)
    x=torch.randint(0,256,(1,2048),device='cuda');other=x.clone();other[:,1024:]=(other[:,1024:]+1)%256
    with torch.no_grad():assert torch.equal(model(x)[:,:1024],model(other)[:,:1024])
    h=torch.randn(1,2048,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    out=model.blocks[0](h,model.cos,model.sin)
    grad,=torch.autograd.grad(out[:,:1024].float().square().sum(),h)
    assert grad[:,1024:].count_nonzero()==0


def test_compiled_and_checkpointed_backward():
    eager=build();activate_branches(eager)
    x=torch.randint(0,256,(2,129),device='cuda');y=(x+1)%256
    losses=[];grads=[]
    for checkpoint in (None,False,True):
        model=eager if checkpoint is None else ConnectionModel(replace(eager.config,activation_checkpoint=checkpoint)).prepare(compile_blocks=True)
        if model is not eager:model.load_state_dict(eager.state_dict())
        loss=model(x,y);loss.backward();losses.append(float(loss))
        grads.append(torch.cat([p.grad.float().flatten() for p in model.parameters()]))
    result=dict(losses=losses,gradients=[relative(g,grads[0]) for g in grads[1:]])
    record('loop-compile',result)
    assert max(losses)-min(losses)<.002 and max(result['gradients'])<.02,result


def test_exact_saved_optimizer_rng_and_next_update(tmp_path):
    model=build();opt,wd=optimizer_for(model)
    x=torch.randint(0,256,(2,129),device='cuda');y=(x+1)%256
    for step in range(3):
        opt.zero_grad(set_to_none=True);schedule(opt,step,305176,wd);model(x,y).backward();opt.step()
    gradients={n:p.grad.clone() for n,p in model.named_parameters()}
    path=tmp_path/'snapshot.pt';torch.save(snapshot(model,opt,3,config=asdict(model.config)),path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    other=ConnectionModel(ConnectionConfig.from_checkpoint(state['config'])).prepare();other_opt,_=optimizer_for(other)
    other.load_state_dict(state['model']);other_opt.load_state_dict(state['optimizer'])
    equal_tree(model.state_dict(),other.state_dict());equal_tree(opt.state_dict(),other_opt.state_dict())
    restore_rng(state['rng']);expected=(torch.rand(8),torch.rand(8,device='cuda'))
    restore_rng(state['rng']);equal_tree(expected,(torch.rand(8),torch.rand(8,device='cuda')))
    for m,o in ((model,opt),(other,other_opt)):
        for n,p in m.named_parameters():p.grad=gradients[n].clone()
        schedule(o,3,305176,wd);o.step()
    equal_tree(model.state_dict(),other.state_dict());equal_tree(opt.state_dict(),other_opt.state_dict())


@pytest.mark.parametrize('options',[dict(attention_loops=0),dict(attention_loop_window=0),dict(attention_kernel='fla_triton'),dict(variant='mhc'),dict(architecture='native'),dict(attention_statistic='variance'),dict(attention_loops=1.5)])
def test_reject_unvalidated_combinations(options):
    with pytest.raises(ValueError):ConnectionConfig(attention_looping=True,**options)


def test_flops_include_two_additional_qkvo_and_windowed_attention():
    c=ConnectionConfig(attention_looping=True);base=model_cost(replace(c,attention_looping=False));loop=model_cost(c)
    t,d,l=c.sequence_length,c.width,c.depth
    pairs=sum(min(i+1,128) for i in range(t))
    extra=2*(8*t*l*d*d+4*d*l*pairs)
    assert loop['estimated_forward_flops']-base['estimated_forward_flops']==extra
    assert loop['estimated_training_flops']-base['estimated_training_flops']==3*extra
    assert model_cost(replace(c,attention_loops=1))['estimated_forward_flops']==base['estimated_forward_flops']


def test_cached_decoding_explicitly_rejected():
    model=build();h=torch.randn(1,129,128,device='cuda',dtype=torch.bfloat16)
    with pytest.raises(ValueError,match='per-loop KV'):
        model.blocks[0](h,model.cos,model.sin,object())
