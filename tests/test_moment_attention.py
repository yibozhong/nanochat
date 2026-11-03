"""Independent weighted-central-moment oracles, live branches, and recovery."""
from dataclasses import asdict

import pytest
import torch

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.moment_attention import value_moments
from nanochat.packed_training import seed_all, snapshot, restore_rng
from nanochat.engine import KVCache
from test_connection_training_resume import equal_tree


@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.recompile_limit = 64
    configure_attention('sdpa_cudnn')
    seed_all(42)


def oracle(q, k, v, centered, offset=0):
    # Deliberately use sum a*(v-mean)^2, NOT E[v^2]-E[v]^2.
    result = []
    for i in range(q.shape[1]):
        keys = k[:, :offset+i+1]
        vals = v[:, :offset+i+1]
        logits = torch.einsum('bhd,bshd->bhs', q[:, i], keys) / q.shape[-1]**.5
        a = logits.softmax(-1).transpose(1, 2).unsqueeze(-1)
        mean = (a*vals).sum(1, keepdim=True)
        result.append((a*(vals-mean if centered else vals).square()).sum(1))
    return torch.stack(result, dim=1)


@pytest.mark.parametrize('centered', [False, True])
def test_fp64_formula_gradients_and_tiling(centered):
    xs = [torch.randn(2, 11, 2, 8, device='cuda', dtype=torch.float64, requires_grad=True) for _ in range(3)]
    expected = oracle(*xs, centered)
    actual = value_moments(*xs, centered=centered, query_chunk=4)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    upstream = torch.randn_like(actual)
    ga = torch.autograd.grad((actual*upstream).sum(), xs, retain_graph=True)
    ge = torch.autograd.grad((expected*upstream).sum(), xs)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize('centered', [False, True])
def test_compiled_fp32_forward_backward(centered):
    xs = [torch.randn(2, 32, 2, 16, device='cuda', requires_grad=True) for _ in range(3)]
    fn = torch.compile(value_moments, fullgraph=True, dynamic=False)
    actual = fn(*xs, centered=centered, query_chunk=16)
    ref_x = [x.detach().double().requires_grad_() for x in xs]
    expected = oracle(*ref_x, centered)
    torch.testing.assert_close(actual.double(), expected, atol=3e-6, rtol=3e-6)
    up = torch.randn_like(actual)
    ga = torch.autograd.grad((actual*up).sum(), xs)
    ge = torch.autograd.grad((expected*up.double()).sum(), ref_x)
    for a,e in zip(ga, ge):
        assert (a.double()-e).norm()/e.norm().clamp_min(1e-12) < 3e-6


def test_variance_not_entropy_constant_values_and_mean_collision():
    q = k = torch.zeros(1, 2, 1, 2, device='cuda', dtype=torch.float64)
    v = torch.tensor([-1.,1.], device='cuda', dtype=torch.float64).view(1,2,1,1).expand_as(q)
    s = value_moments(q,k,v,centered=True)
    torch.testing.assert_close(s[:,0], torch.zeros_like(s[:,0]), atol=0, rtol=0)
    torch.testing.assert_close(s[:,1], torch.ones_like(s[:,1]), atol=0, rtol=0)
    torch.testing.assert_close(value_moments(q,k,torch.ones_like(v)*3,centered=True),torch.zeros_like(v),atol=0,rtol=0)


def build(mode, *, compiled=False, length=128):
    seed_all(42)
    c = ConnectionConfig(depth=2,width=128,heads=1,vocab_size=256,sequence_length=length,
                         attention_statistic=mode, moment_query_chunk=128)
    return ConnectionModel(c).prepare(compile_blocks=compiled)


@pytest.mark.parametrize('mode', ['second_moment', 'variance'])
def test_matched_initialization_zero_branch_and_optimizer(mode):
    base = build('none'); before = torch.get_rng_state().clone()
    model = build(mode)
    assert torch.equal(before,torch.get_rng_state())
    a,b=base.state_dict(),model.state_dict()
    assert all(torch.equal(v,b[k]) for k,v in a.items())
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in base.parameters())==2*128**2
    x = torch.randint(0,256,(2,128),device='cuda')
    torch.testing.assert_close(model(x),base(x),rtol=0,atol=0)
    model(x,(x+1)%256).backward()
    for block in model.blocks:
        assert block.attn.c_stat.weight.count_nonzero()==0
        assert block.attn.c_stat.weight.grad.norm()>0
    opt,wd=optimizer_for(model); original,old_wd=optimizer_for(base)
    assert wd==old_wd
    assert all(any(p is b.attn.c_stat.weight for g in opt.param_groups if g['kind']=='muon' for p in g['params']) for b in model.blocks)
    opt.step()
    assert all(b.attn.c_stat.weight.norm()>0 for b in model.blocks)


def activate(model):
    with torch.no_grad():
        for block in model.blocks:
            for weight in (block.attn.c_proj.weight,block.attn.c_stat.weight,block.mlp.c_proj.weight):
                weight.normal_(std=.02)


@pytest.mark.parametrize('mode', ['second_moment', 'variance'])
def test_live_2k_causality_and_cache(mode):
    model=build(mode,length=2048);activate(model);model.eval()
    x=torch.randint(0,256,(1,2048),device='cuda')
    changed=x.clone();changed[:,1024:]=(changed[:,1024:]+7)%256
    with torch.no_grad():
        full=model(x)
        torch.testing.assert_close(model(changed)[:,:1024],full[:,:1024],rtol=0,atol=0)
        cache=KVCache(1,1,2048,128,2,'cuda',torch.bfloat16)
        parts=[model(x[:,:1024],kv_cache=cache),model(x[:,1024:1536],kv_cache=cache),model(x[:,1536:],kv_cache=cache)]
        torch.testing.assert_close(torch.cat(parts,1),full,atol=.004,rtol=.02)
        assert cache.get_pos()==2048
    # Future inputs also have exactly zero VJP influence on prefix predictions.
    embedding=model.wte(x).detach().requires_grad_()
    handle=model.wte.register_forward_hook(lambda *_:embedding)
    try:
        model(x)[:,:1024].float().square().sum().backward()
        assert embedding.grad[:,1024:].count_nonzero()==0
    finally:handle.remove()


@pytest.mark.parametrize('mode', ['second_moment', 'variance'])
def test_live_compiled_loss_gradients_and_recompute(mode):
    eager=build(mode);activate(eager)
    compiled=build(mode,compiled=True);compiled.load_state_dict(eager.state_dict())
    x=torch.randint(0,256,(2,128),device='cuda');y=(x+1)%256
    losses=[];grads=[]
    for model in (eager,compiled):
        loss=model(x,y);loss.backward();losses.append(loss.detach().float())
        grads.append({n:p.grad.detach().float().clone() for n,p in model.named_parameters()})
    assert abs(float(losses[0]-losses[1]))<.002
    squared_error=sum((grads[0][n]-grads[1][n]).square().sum() for n in grads[0])
    squared_norm=sum(v.square().sum() for v in grads[0].values())
    assert (squared_error/squared_norm).sqrt()<.02
    for n in grads[0]:
        diff=(grads[0][n]-grads[1][n]).norm()
        assert diff <= 1e-6 + .03*grads[0][n].norm(), (n,float(diff),float(grads[0][n].norm()))
    compiled.zero_grad(set_to_none=True);compiled.config.activation_checkpoint=True
    other=compiled(x,y);other.backward()
    torch.testing.assert_close(other,losses[1],atol=1e-6,rtol=1e-6)
    for n,p in compiled.named_parameters():
        assert (p.grad.float()-grads[1][n]).norm() <= 1e-6 + .02*grads[1][n].norm()


@pytest.mark.parametrize('mode', ['second_moment', 'variance'])
def test_checkpoint_optimizer_rng_and_same_gradient_next_update(mode,tmp_path):
    model=build(mode);opt,wd=optimizer_for(model)
    x=torch.randint(0,256,(2,128),device='cuda');y=(x+1)%256
    for step in range(3):
        opt.zero_grad(set_to_none=True);schedule(opt,step,305176,wd)
        model(x,y).backward();opt.step()
    grads={n:p.grad.clone() for n,p in model.named_parameters()}
    path=tmp_path/'state.pt';torch.save(snapshot(model,opt,3,config=asdict(model.config)),path)
    loaded=torch.load(path,map_location='cpu',weights_only=False)
    other=build(mode);other_opt,other_wd=optimizer_for(other)
    other.load_state_dict(loaded['model']);other_opt.load_state_dict(loaded['optimizer'])
    equal_tree(model.state_dict(),other.state_dict());equal_tree(opt.state_dict(),other_opt.state_dict())
    restore_rng(loaded['rng']);expected=(torch.rand(9),torch.rand(9,device='cuda'))
    restore_rng(loaded['rng']);equal_tree(expected,(torch.rand(9),torch.rand(9,device='cuda')))
    for target,optimizer in ((model,opt),(other,other_opt)):
        for n,p in target.named_parameters():p.grad=grads[n].clone()
        schedule(optimizer,3,305176,wd);optimizer.step()
    equal_tree(model.state_dict(),other.state_dict());equal_tree(opt.state_dict(),other_opt.state_dict())


def test_new_config_is_explicit_old_checkpoint_defaults_and_scope():
    assert ConnectionConfig.from_checkpoint({'variant':'baseline'}).attention_statistic=='none'
    for changes in ({'variant':'mhc'},{'architecture':'native'}):
        with pytest.raises(ValueError):ConnectionConfig(attention_statistic='variance',**changes)
