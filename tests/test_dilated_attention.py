"""Independent sparse-mask/VJP, full-model and lifecycle validation on GPU."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import pytest
import torch
from nanochat.dilated_pattern import dilation_schedule,selected_positions
from nanochat.dilated_attention import sparse_attention,sparse_reference
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for,schedule
from nanochat.connection_experiment import configure_attention
from nanochat.packed_training import seed_all,snapshot,restore_rng
from nanochat.engine import KVCache
from test_connection_training_resume import equal_tree


def relative(a,b):return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()


def record(name,value):
    root=Path('results/dilated-checks');root.mkdir(parents=True,exist_ok=True)
    (root/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


@pytest.fixture(autouse=True)
def fresh_cache():
    torch._dynamo.reset();torch.backends.cuda.matmul.allow_tf32=False
    yield
    torch._dynamo.reset()


def oracle(q,k,v,stride,core,remote,sinks):
    # Independent explicit set construction, rather than the GPU modular mask.
    t=q.shape[1];mask=torch.zeros(t,t,dtype=torch.bool)
    for row in range(t):
        allowed=set(range(min(sinks,row+1)))
        allowed.update(range(max(0,row-core+1),row+1))
        for offset in range(remote):
            key=row-core-offset*stride
            if key>=0:allowed.add(key)
        mask[row,list(allowed)]=True
    mask=mask.to(q.device)
    scores=torch.einsum('bihd,bjhd->bhij',q,k)*q.shape[-1]**-.5
    return torch.einsum('bhij,bjhd->bihd',scores.masked_fill(~mask,-float('inf')).softmax(-1),v)


CASES=([(1,1,32,1,128,124,4,'random'),(17,2,64,3,8,7,4,'random'),
       (129,2,128,5,64,60,4,'random'),(257,2,64,7,64,64,8,'random')]+
      [(2048,4,128,s,128,124,4,'random') for s in (1,3,5,7,9,11,14)]+
      [(131,2,64,3,64,60,4,m) for m in ('zero','tiny','large')])


@pytest.mark.parametrize('t,h,d,stride,core,remote,sinks,mode',CASES)
def test_primitive_all_vjps(t,h,d,stride,core,remote,sinks,mode):
    torch.manual_seed(8242);b=2 if t<2048 else 1
    inputs=[torch.randn(b,t,h,d,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    if mode=='zero':
        for x in inputs:x.zero_()
    elif mode=='tiny':
        for x in inputs:x.mul_(1e-5)
    elif mode=='large':
        inputs[0].mul_(3);inputs[1].mul_(3)
    for x in inputs:x.requires_grad_()
    before=[x.detach().clone() for x in inputs]
    refs=[x.detach().double().requires_grad_() for x in inputs];go=torch.randn_like(inputs[0])
    expected=oracle(*refs,stride,core,remote,sinks);eg=torch.autograd.grad(expected,refs,go.double())
    fn=torch.compile(sparse_attention,fullgraph=True,dynamic=False)
    observed=fn(*inputs,stride,core,remote,sinks);og=torch.autograd.grad(observed,inputs,go)
    errors=dict(output=relative(observed,expected),gradients=[relative(a,b) for a,b in zip(og,eg)])
    assert errors['output']<.008,errors
    assert max(errors['gradients'])<.02,errors
    assert all(torch.isfinite(x).all() for x in [observed,*og])
    assert all(torch.equal(x,y) for x,y in zip(inputs,before))
    record(f'primitive-{t}-{h}-{d}-{stride}-{core}-{mode}',errors)


def test_schedule_geometry_and_cost():
    from nanochat.connection_cost import model_cost
    strides=dilation_schedule(2048,8)
    assert strides==(1,3,5,7,9,11,14,None)
    assert dilation_schedule(2048,8,progressive=False)==(1,)*7+(None,)
    for t in (17,257,2048):
        c=ConnectionConfig(attention_pattern='progressive_dilated',streams=1,sequence_length=t)
        plan=dilation_schedule(t,c.depth);count=0
        for stride in plan:
            if stride is None:count+=t*(t+1)//2;continue
            for i in range(t):
                positions=selected_positions(i,stride,128,124,4)
                assert len(positions)<=256 and positions[-1]==i
                assert set(range(min(4,i+1)))<=set(positions)
                assert set(range(max(0,i-127),i+1))<=set(positions)
                count+=len(positions)
        assert model_cost(c)['components_forward']['attention']==4*c.width*count
    record('geometry',dict(strides=strides,total_budget=256,core=128,sinks=4,remote=124))


def test_double_reference_gradcheck():
    xs=[torch.randn(1,7,1,32,device='cuda',dtype=torch.float64,requires_grad=True) for _ in range(3)]
    f=lambda *x:sparse_reference(*x,3,2,3,1)
    assert torch.autograd.gradcheck(f,tuple(xs),fast_mode=True,eps=1e-6,atol=1e-5,rtol=1e-4)
    torch.testing.assert_close(f(*xs),oracle(*xs,3,2,3,1),rtol=1e-12,atol=1e-12)


def build(pattern='progressive_dilated',backend='triton',compiled=False,nonzero=True):
    configure_attention('hardware_auto');seed_all(42)
    m=ConnectionModel(ConnectionConfig(variant='baseline',streams=1,depth=3,width=128,heads=2,
        vocab_size=128,sequence_length=2048,attention_pattern=pattern,sparse_backend=backend)).prepare(compile_blocks=compiled)
    if nonzero:
        with torch.no_grad():
            for b in m.blocks:
                b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
    return m


def test_identity_rng_parameters_and_final_full():
    m=build(nonzero=False);state=torch.get_rng_state();baseline=build('full',nonzero=False)
    assert torch.equal(state,torch.get_rng_state());equal_tree(m.state_dict(),baseline.state_dict())
    assert type(m.blocks[-1].attn) is type(baseline.blocks[-1].attn)
    assert ConnectionConfig.from_checkpoint(asdict(m.config))==m.config
    with torch.device('meta'):
        full=ConnectionModel(ConnectionConfig(streams=1,attention_pattern='progressive_dilated'))
    assert sum(p.numel() for p in full.parameters())==58720256
    a,_=optimizer_for(m);b,_=optimizer_for(baseline)
    assert [len(g['params']) for g in a.param_groups]==[len(g['params']) for g in b.param_groups]
    record('identity',dict(extra_parameters=0,total_parameters=58720256))


def test_fullmodel_all_gradients():
    ref=build(backend='torch');actual=build(compiled=True)
    ids=torch.randint(0,128,(2,257),device='cuda');targets=(ids+1)%128
    expected=ref(ids,targets);expected.backward();observed=actual(ids,targets);observed.backward()
    errors={n:relative(p.grad,dict(ref.named_parameters())[n].grad) for n,p in actual.named_parameters()}
    torch.testing.assert_close(expected,observed,atol=2e-4,rtol=2e-4)
    assert max(errors.values())<.03,errors
    record('model-gradients',dict(loss_difference=(expected-observed).abs().item(),gradients=errors))


@pytest.mark.parametrize('pattern',['sink_swa','progressive_dilated'])
def test_model_causal_cache_cce_recompute(pattern):
    m=build(pattern,compiled=True);ids=torch.randint(0,128,(2,2048),device='cuda');changed=ids.clone()
    changed[0,1024:]=(changed[0,1024:]+19)%128
    with torch.no_grad():
        first=m(ids);other=m(changed)
        assert torch.equal(first[0,:1024],other[0,:1024]) and torch.equal(first[1],other[1])
        compiled=m.compiled_steps;m.compiled_steps=None
        cache=KVCache(2,2,512,64,3,'cuda',torch.bfloat16)
        pre=m(ids[:,:257],kv_cache=cache);chunk=m(ids[:,257:260],kv_cache=cache);one=m(ids[:,260:261],kv_cache=cache)
        whole=m(ids[:,:261])
        torch.testing.assert_close(pre,whole[:,:257],atol=.004,rtol=.03)
        torch.testing.assert_close(chunk,whole[:,257:260],atol=.004,rtol=.03)
        torch.testing.assert_close(one,whole[:,260:261],atol=.004,rtol=.03)
        assert cache.get_pos()==261
        m.compiled_steps=compiled
    x,y=ids[:,:257].contiguous(),ids[:,1:258].contiguous()
    sparse=m(x,y);dense=m(x,y,dense=True)
    torch.testing.assert_close(sparse,dense,atol=.003,rtol=.001)
    m.zero_grad(set_to_none=True);sparse.backward();grads={n:p.grad.clone() for n,p in m.named_parameters()}
    m.zero_grad(set_to_none=True);m.config.activation_checkpoint=True
    loss=m(x,y);loss.backward()
    assert max(relative(p.grad,grads[n]) for n,p in m.named_parameters())<.005
    record('lifecycle-'+pattern,dict(causal=True,cache=True,cce=True,recompute=True))


def test_optimizer_rng_restore():
    m=build();opt,wd=optimizer_for(m)
    schedule(opt,0,305176,wd)
    for p in m.parameters():p.grad=torch.full_like(p,.001)
    opt.step();opt.zero_grad(set_to_none=True)
    state=copy.deepcopy(snapshot(m,opt,1))
    expected_random=torch.rand(9,device='cuda')
    restored=build();other,other_wd=optimizer_for(restored)
    restored.load_state_dict(state['model']);other.load_state_dict(state['optimizer']);restore_rng(state['rng'])
    assert torch.equal(expected_random,torch.rand(9,device='cuda'))
    for model,optimizer,decay in [(m,opt,wd),(restored,other,other_wd)]:
        schedule(optimizer,1,305176,decay)
        for p in model.parameters():p.grad=torch.full_like(p,.002)
        optimizer.step()
    equal_tree(m.state_dict(),restored.state_dict());equal_tree(opt.state_dict(),other.state_dict())


def test_eight_layer_train_eval_compile_budget():
    configure_attention('hardware_auto');seed_all(42)
    model=ConnectionModel(ConnectionConfig(streams=1,depth=8,width=128,heads=2,
        vocab_size=128,sequence_length=2048,attention_pattern='progressive_dilated')).prepare(compile_blocks=True)
    for batch in (2,1):
        x=torch.randint(0,128,(batch,129),device='cuda');y=(x+1)%128
        model.train();model.zero_grad(set_to_none=True)
        loss=model(x,y);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
        model.eval()
        with torch.no_grad():
            evaluated=model(x,y)
        torch.testing.assert_close(loss,evaluated,atol=.003,rtol=.001)
    record('eight-layer-compile',dict(train_eval=True,full_and_remainder_batches=True))


@pytest.mark.parametrize('options',[dict(architecture='native'),dict(streams=4),dict(attention_kernel='fla_triton'),
    dict(sparse_budget=133),dict(depth=1),dict(sparse_backend='bad'),dict(sparse_sinks=0),dict(variant='mhc')])
def test_bad_config(options):
    values=dict(attention_pattern='progressive_dilated',streams=1);values.update(options)
    with pytest.raises(ValueError):ConnectionConfig(**values)
