"""Alternating attention replacement, gradients, causality and training state."""
from dataclasses import asdict,replace
import pytest
import torch
from torch.nn import functional as F
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for,schedule
from nanochat.connection_experiment import configure_attention
from nanochat.connection_cost import model_cost
from nanochat.gpt import norm,apply_rotary_emb
from nanochat.engine import KVCache
from nanochat.packed_training import seed_all,snapshot,restore_rng
from test_connection_training_resume import equal_tree
from test_triton_moments import relative,record


def dense_attention(q,k,v):
    q,k,v=(x.transpose(1,2) for x in (q,k,v))
    t=q.shape[-2];mask=torch.ones(t,t,device=q.device,dtype=torch.bool).tril()
    p=((q.float()@k.float().transpose(-1,-2))*q.shape[-1]**-.5).masked_fill(~mask,-torch.inf).softmax(-1)
    return (p@v.float()).to(v.dtype).transpose(1,2)

@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(2);seed_all(42);torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.recompile_limit=64;configure_attention('fa3')


def build(**kwargs):
    opts=dict(depth=4,width=128,heads=1,sequence_length=129,vocab_size=256,mlp_heavy=True)
    opts.update(kwargs)
    return ConnectionModel(ConnectionConfig(**opts)).prepare()


def activate(model):
    with torch.no_grad():
        for b in model.blocks:
            (b.attn.c_proj if b.attn is not None else b.attention_mlp.c_proj).weight.normal_(std=.02)
            b.mlp.c_proj.weight.normal_(std=.02)


def test_full_size_topology_parameters_initialization_rng():
    seed_all(42);base=ConnectionModel(ConnectionConfig()).prepare();rng=torch.get_rng_state().clone()
    seed_all(42);model=ConnectionModel(ConnectionConfig(mlp_heavy=True)).prepare()
    assert torch.equal(rng,torch.get_rng_state())
    assert sum(p.numel() for p in base.parameters())==58720256
    assert sum(p.numel() for p in model.parameters())==62914560
    assert [i for i,b in enumerate(model.blocks) if b.attn is None]==[1,3,5,7]
    assert [b.attn.layer_idx for b in model.blocks if b.attn is not None]==[0,1,2,3]
    old=base.state_dict();state=model.state_dict()
    for name,p in state.items():
        if 'attention_mlp' not in name:assert torch.equal(p,old[name]),name
    for i in (1,3,5,7):
        b=model.blocks[i];assert b.attention_mlp.c_fc.weight.shape==(2048,512)
        assert b.attention_mlp.c_proj.weight.count_nonzero()==0
        assert b.attention_mlp.c_fc.weight.data_ptr()!=b.mlp.c_fc.weight.data_ptr()
        assert not torch.equal(b.attention_mlp.c_fc.weight,b.mlp.c_fc.weight)
        assert not any(n.startswith(f'blocks.{i}.attn.') for n in state)
    legacy=asdict(base.config);legacy.pop('mlp_heavy');assert not ConnectionConfig.from_checkpoint(legacy).mlp_heavy
    x=torch.randint(0,32768,(1,128),device='cuda')
    with torch.no_grad():assert torch.equal(base(x),model(x))


def test_double_mlp_fp64_formula_and_gradients():
    model=build();block=model.blocks[1].double()
    with torch.no_grad():
        block.attention_mlp.c_proj.weight.normal_(std=.02);block.mlp.c_proj.weight.normal_(std=.02)
    x=torch.randn(2,17,128,device='cuda',dtype=torch.float64,requires_grad=True)
    weights=[p for p in block.parameters()];refx=x.detach().clone().requires_grad_();refs=[p.detach().clone().requires_grad_() for p in weights]
    byname=dict(zip([n for n,p in block.named_parameters()],refs))
    def independent_mlp(h,prefix):
        # Explicit RMS, two affine transforms and ReLU squared, outside Block.
        h=h*(h.square().mean(-1,keepdim=True)+torch.finfo(h.dtype).eps).rsqrt()
        return F.linear(F.relu(F.linear(h,byname[prefix+'.c_fc.weight'])).square(),byname[prefix+'.c_proj.weight'])
    z=refx+independent_mlp(refx,'attention_mlp');expected=z+independent_mlp(z,'mlp')
    actual=block(x,None,None);g=torch.randn_like(x)
    ag=torch.autograd.grad((actual*g).sum(),[x,*weights]);eg=torch.autograd.grad((expected*g).sum(),[refx,*refs])
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
    for a,b in zip(ag,eg):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-9)


def test_full_model_manual_dense_oracle_and_parameter_gradients():
    model=build();activate(model);reference=build();reference.load_state_dict(model.state_dict())
    idx=torch.randint(0,256,(2,129),device='cuda');h=norm(reference.wte(idx));b,t,d=h.shape
    for block in reference.blocks:
        if block.attn is not None:
            a=block.attn;n=norm(h);shape=(b,t,a.n_head,a.head_dim)
            q=norm(apply_rotary_emb(a.c_q(n).view(shape),reference.cos,reference.sin))*1.2
            k=norm(apply_rotary_emb(a.c_k(n).view(shape),reference.cos,reference.sin))*1.2
            v=a.c_v(n).view(shape)
            update=a.c_proj(dense_attention(q,k,v).reshape(b,t,d))
        else:update=block.attention_mlp(norm(h))
        h=h+update;h=h+block.mlp(norm(h))
    expected=norm(h);actual=model.hidden(idx);g=torch.randn_like(actual)
    (actual.float()*g.float()).sum().backward();(expected.float()*g.float()).sum().backward()
    grads=torch.cat([p.grad.float().flatten() for p in model.parameters() if p.grad is not None]);refs=torch.cat([p.grad.float().flatten() for p in reference.parameters() if p.grad is not None])
    result=dict(hidden=relative(actual,expected),gradients=relative(grads,refs));record('mlp-heavy-manual',result)
    assert result['hidden']<.025 and result['gradients']<.035,result
    assert all(p.grad is not None and p.grad.norm()>0 for block in model.blocks for p in block.parameters())


def test_compile_and_activation_checkpoint():
    eager=build();activate(eager);idx=torch.randint(0,256,(2,129),device='cuda');y=(idx+1)%256
    losses=[];grads=[]
    for cp in (None,False,True):
        model=eager if cp is None else ConnectionModel(replace(eager.config,activation_checkpoint=cp)).prepare(compile_blocks=True)
        if model is not eager:model.load_state_dict(eager.state_dict())
        loss=model(idx,y);loss.backward();losses.append(float(loss))
        grads.append(torch.cat([p.grad.float().flatten() for p in model.parameters()]))
    result=dict(losses=losses,gradient_relative=[relative(g,grads[0]) for g in grads[1:]]);record('mlp-heavy-compile',result)
    assert max(losses)-min(losses)<.002 and max(result['gradient_relative'])<.02,result


def test_real_2k_causality_and_tokenwise_replacement():
    model=build(sequence_length=2048);activate(model)
    x=torch.randint(0,256,(1,2048),device='cuda');other=x.clone();other[:,1024:]=(other[:,1024:]+1)%256
    with torch.no_grad():assert torch.equal(model(x)[:,:1024],model(other)[:,:1024])
    h=torch.randn(1,2048,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    out=model.blocks[1](h,model.cos,model.sin);g,=torch.autograd.grad(out[:,777].float().square().sum(),h)
    assert g[:,:777].count_nonzero()==0 and g[:,778:].count_nonzero()==0 and g[:,777].norm()>0


@pytest.mark.parametrize('depth',[3,4])
def test_kv_cache_slot_count_and_position_advance(depth):
    model=build(depth=depth,sequence_length=257).eval();activate(model)
    idx=torch.randint(0,256,(1,133),device='cuda')
    def cache(n):return KVCache(1,1,257,128,n,'cuda',torch.bfloat16)
    kv=cache((depth+1)//2)
    with torch.no_grad():
        expected=model(idx);pieces=[]
        for start,stop in [(0,128),(128,132),(132,133)]:
            pieces.append(model(idx[:,start:stop],kv_cache=kv));assert kv.get_pos()==stop
        actual=torch.cat(pieces,1)
        with pytest.raises(ValueError,match='one slot per retained attention'):model(idx[:,:1],kv_cache=cache(depth))
    error=relative(actual,expected);record(f'mlp-heavy-cache-{depth}',dict(relative=error))
    assert error<.015


def test_exact_optimizer_rng_checkpoint_and_fixed_gradient_update(tmp_path):
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


def test_cce_dense_loss_and_gradient_agree():
    model=build();activate(model);other=build();other.load_state_dict(model.state_dict())
    x=torch.randint(0,256,(2,129),device='cuda');y=(x+1)%256
    a=model(x,y);b=other(x,y,dense=True);a.backward();b.backward()
    g=torch.cat([p.grad.float().flatten() for p in model.parameters()]);r=torch.cat([p.grad.float().flatten() for p in other.parameters()])
    result=dict(losses=[float(a),float(b)],gradient_relative=relative(g,r));record('mlp-heavy-cce',result)
    assert abs(float(a)-float(b))<.002 and result['gradient_relative']<.02,result


def test_optimizer_coverage_and_original_hyperparameters():
    model=build();base=build(mlp_heavy=False);opt,wd=optimizer_for(model);baseline,bwd=optimizer_for(base);assert wd==bwd
    ids=[id(p) for group in opt.param_groups for p in group['params']]
    assert len(ids)==len(set(ids))==len(list(model.parameters()))
    for block in model.blocks:
        if block.attention_mlp is None:continue
        for parameter in block.attention_mlp.parameters():
            group=next(g for g in opt.param_groups if any(p is parameter for p in g['params']))
            original=next(g for g in baseline.param_groups if g['kind']=='muon' and g['params'][0].shape==parameter.shape)
            equal_tree({k:v for k,v in group.items() if k!='params'},{k:v for k,v in original.items() if k!='params'})


@pytest.mark.parametrize('opts',[dict(architecture='native'),dict(variant='mhc'),dict(attention_kernel='fla_triton'),dict(attention_statistic='variance')])
def test_unsupported_combinations_rejected(opts):
    with pytest.raises(ValueError):ConnectionConfig(mlp_heavy=True,**opts)


def test_cost_counts_four_attention_and_twelve_mlp():
    c=ConnectionConfig(mlp_heavy=True);result=model_cost(c);t,d,v=c.sequence_length,c.width,c.vocab_size
    expected=(4*4+12*8)*2*t*d*d+4*4*d*(t*(t+1)//2)+2*t*d*v
    assert result['estimated_forward_flops']==expected
    assert result['estimated_training_flops']==3*expected
    base=model_cost(replace(c,mlp_heavy=False))['estimated_forward_flops']
    assert abs(expected/base-1)<.001


def test_legacy_inactive_loop_metadata_and_retired_loop_rejection():
    config=asdict(ConnectionConfig());config.update(attention_looping=False,attention_loops=3,attention_loop_window=128)
    assert ConnectionConfig.from_checkpoint(config)==ConnectionConfig()
    config['attention_looping']=True
    with pytest.raises(ValueError,match='retired attention looping'):
        ConnectionConfig.from_checkpoint(config)
