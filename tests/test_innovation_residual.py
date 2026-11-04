"""Gradient isolation, causal evaluation, compiled execution and checkpoint state."""
import copy
from dataclasses import asdict
import pytest
import torch
from torch.nn import functional as F

from nanochat.innovation_residual import InnovationResidual, prediction_sums, summarize_prediction_sums
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.packed_training import seed_all, snapshot, restore_rng, evaluate
from test_connection_training_resume import equal_tree


def build(enabled=True, compiled=False, checkpoint=False, length=128):
    seed_all(42)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.recompile_limit=64
    return ConnectionModel(ConnectionConfig(innovation_residual=enabled, attention_kernel='fla_triton',
        depth=2,width=128,heads=1,vocab_size=256,sequence_length=length,
        activation_checkpoint=checkpoint)).prepare(compile_blocks=compiled)


def test_fp64_formula_and_both_gradient_firewalls():
    seed_all(42)
    p=InnovationResidual(8).cuda().double()
    with torch.no_grad():p.weight.normal_(); p.bias.normal_()
    h=torch.randn(2,5,8,device='cuda',dtype=torch.float64,requires_grad=True)
    r=torch.randn_like(h,requires_grad=True); beta=h.new_tensor(.6)
    z,local,_=p(h,r,beta,True)
    expected=h+r-beta*F.linear(h,p.weight,p.bias)
    torch.testing.assert_close(z,expected)
    probe=torch.randn_like(z)
    gh,gr,gw,gb=torch.autograd.grad((z*probe).sum(),(h,r,p.weight,p.bias),allow_unused=True,retain_graph=True)
    torch.testing.assert_close(gh,probe-beta*(probe@p.weight))
    torch.testing.assert_close(gr,probe)
    assert gw is None and gb is None
    gh,gr,gw,gb=torch.autograd.grad(local,(h,r,p.weight,p.bias),allow_unused=True)
    assert gh is None and gr is None
    # Local MSE deliberately reduces in FP32, including in this FP64 oracle.
    oracle=F.mse_loss(F.linear(h.detach(),p.weight,p.bias).float(),r.detach().float())
    ew,eb=torch.autograd.grad(oracle,(p.weight,p.bias))
    torch.testing.assert_close(gw,ew); torch.testing.assert_close(gb,eb)


def test_initialization_beta_zero_and_rng_match():
    base=build(False); rng=torch.get_rng_state().clone(); state=base.state_dict()
    model=build(); assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    for block in model.blocks:
        for p in (block.attn_innovation,block.mlp_innovation):
            torch.testing.assert_close(p.weight,torch.eye(128,device='cuda')*.001,rtol=0,atol=0)
            assert p.bias.count_nonzero()==0
    x=torch.randint(256,(2,128),device='cuda')
    torch.testing.assert_close(base(x),model(x),rtol=0,atol=0)


@pytest.mark.parametrize('options',[dict(architecture='native'),dict(variant='mhc'),dict(attention_statistic='variance')])
def test_reject_uncontrolled_combinations(options):
    with pytest.raises(ValueError):ConnectionConfig(innovation_residual=True,**options)


def test_r_squared_full_dataset_centering_and_undefined_constant():
    torch.manual_seed(21)
    r=torch.randn(2,9,8,device='cuda');p=.7*r+.2;h=r+1;innovation=r-p
    a=prediction_sums(h,r,p,innovation)
    b=prediction_sums(h[:1],r[:1],p[:1],innovation[:1])+prediction_sums(h[1:],r[1:],p[1:],innovation[1:])
    torch.testing.assert_close(a,b,rtol=1e-14,atol=1e-14)
    result=summarize_prediction_sums(torch.stack([a,a]),8)[0]
    rf=r.double().flatten(0,1);pf=p.double().flatten(0,1)
    oracle=1-(pf-rf).square().sum()/(rf-rf.mean(0)).square().sum()
    assert abs(result['r_squared']-oracle.item())<1e-12
    r=torch.ones_like(r)
    a=prediction_sums(h,r,p,r-p)
    assert summarize_prediction_sums(torch.stack([a,a]),8)[0]['r_squared'] is None
    a=prediction_sums(h,h,10*h,-9*h)
    assert summarize_prediction_sums(torch.stack([a,a]),8)[0]['r_squared']<0


def test_full_model_objectives_update_disjoint_parameters():
    model=build();model.set_innovation_step(39)
    x=torch.randint(256,(2,128),device='cuda'); y=(x+1)%256
    ce,local,_=model(x,y,return_innovation=True)
    ce.backward(retain_graph=True)
    assert all(p.grad is None for n,p in model.named_parameters() if '_innovation.' in n)
    assert any(p.grad is not None and p.grad.abs().max()>0 for n,p in model.named_parameters() if '_innovation.' not in n)
    model.zero_grad(set_to_none=True);local.backward()
    assert all(p.grad is None for n,p in model.named_parameters() if '_innovation.' not in n)
    assert all(p.grad is not None for n,p in model.named_parameters() if '_innovation.' in n)
    assert sum(p.grad.float().square().sum() for n,p in model.named_parameters() if '_innovation.' in n)>0


def test_optimizer_coverage_and_baseline_groups_unchanged():
    base=build(False);ob,wd=optimizer_for(base)
    model=build();opt,newwd=optimizer_for(model)
    assert wd==newwd
    groups=[g for g in opt.param_groups if not g.get('local_predictor')]
    assert len(groups)==len(ob.param_groups)
    for a,b in zip(groups,ob.param_groups):
        equal_tree({k:v for k,v in a.items() if k!='params'},{k:v for k,v in b.items() if k!='params'})
        assert [p.shape for p in a['params']]==[p.shape for p in b['params']]
    predictors=[g for g in opt.param_groups if g.get('local_predictor')]
    assert len(predictors)==1 and predictors[0]['kind']=='adamw' and predictors[0]['weight_decay']==0
    assert {id(p) for p in predictors[0]['params']}=={id(p) for n,p in model.named_parameters() if '_innovation.' in n}


def test_warmup_tensor_is_saved_and_does_not_affect_backbone_schedule():
    model=build();opt,wd=optimizer_for(model)
    for step,expected in ((0,0.),(19,19/39),(39,1.),(40,1.),(305175,1.)):
        model.set_innovation_step(step)
        assert model.innovation_beta.item()==pytest.approx(expected)
    assert 'innovation_beta' in model.state_dict()
    schedule(opt,0,305176,wd)
    assert next(g for g in opt.param_groups if g.get('local_predictor'))['lr']==.001/40
    old=asdict(model.config);old.pop('innovation_residual');old.pop('innovation_warmup_steps');old.pop('innovation_predictor_lr')
    assert not ConnectionConfig.from_checkpoint(old).innovation_residual


@pytest.mark.parametrize('checkpoint',[False,True])
def test_compiled_complete_step_matches_eager(checkpoint):
    model=build();other=build(compiled=True,checkpoint=checkpoint)
    with torch.no_grad():
        for block in model.blocks:
            block.attn.c_proj.weight.normal_(std=.01);block.mlp.c_proj.weight.normal_(std=.01)
    model.set_innovation_step(20);other.load_state_dict(model.state_dict())
    x=torch.randint(256,(2,128),device='cuda');y=(x+1)%256
    losses=[];grads=[]
    for m in (model,other):
        ce,aux,_=m(x,y,return_innovation=True);(ce+aux).backward()
        losses.append(torch.stack([ce,aux]));grads.append(torch.cat([p.grad.float().flatten() for p in m.parameters()]))
    torch.testing.assert_close(losses[0],losses[1],rtol=.004,atol=1e-4)
    assert (grads[0]-grads[1]).norm()/grads[0].norm()<.015


def test_2k_causality_and_sample_isolation():
    model=build(length=2048).eval(); model.set_innovation_step(39)
    with torch.no_grad():
        for block in model.blocks:
            block.attn.c_proj.weight.normal_(std=.01);block.mlp.c_proj.weight.normal_(std=.01)
        x=torch.randint(256,(2,2048),device='cuda');changed=x.clone()
        changed[0,1024:]=(changed[0,1024:]+1)%256; changed[1]=(changed[1]+1)%256
        a=model(x);b=model(changed)
        torch.testing.assert_close(a[0,:1024],b[0,:1024],rtol=0,atol=0)


def test_validation_metrics_do_not_train_predictors():
    model=build(compiled=True); model.set_innovation_step(39)
    x=torch.randint(256,(3,128),device='cuda');y=(x+1)%256
    class Data:
        arrays={'dev':range(3)}
        def batch(self,split,start,n):return x[start:start+n],y[start:start+n]
    before=copy.deepcopy(model.state_dict());report={}
    ce,count=evaluate(model,Data(),microbatch=2,innovation_report=report)
    equal_tree(before,model.state_dict());assert all(p.grad is None for p in model.parameters())
    assert model.training and count==384 and ce>0 and len(report['branches'])==4
    assert all(r['tokens']==384 for r in report['branches'])
    assert report['beta']==1.


def test_production_shape_compiled_r_squared_reduction():
    """The old inline FP64 reduction compiled at small sizes but failed here."""
    seed_all(42);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    model=ConnectionModel(ConnectionConfig(innovation_residual=True,attention_kernel='fla_triton',
        depth=1,width=512,heads=4,vocab_size=256,sequence_length=2048)).prepare(compile_blocks=True).eval()
    model.set_innovation_step(39)
    with torch.no_grad():
        for block in model.blocks:
            block.attn.c_proj.weight.normal_(std=.01);block.mlp.c_proj.weight.normal_(std=.01)
        x=torch.randint(256,(16,2048),device='cuda')
        ordinary=model(x)
        observed,_,sums=model(x,return_innovation=True,collect_statistics=True)
        torch.testing.assert_close(ordinary,observed,rtol=.008,atol=1e-4)
        assert sums.shape==(2,1029) and sums.dtype==torch.float64
        assert torch.isfinite(sums).all()
        rows=summarize_prediction_sums(sums,512)
        assert all(r['tokens']==32768 and r['r_squared'] is not None for r in rows)
        assert all(p.grad is None for p in model.parameters())


def test_exact_optimizer_rng_and_next_fixed_gradient_resume(tmp_path):
    model=build();opt,wd=optimizer_for(model)
    x=torch.randint(256,(2,128),device='cuda');y=(x+1)%256
    for step in range(3):
        model.set_innovation_step(step);schedule(opt,step,305176,wd);opt.zero_grad(set_to_none=True)
        ce,aux,_=model(x,y,return_innovation=True);(ce+aux).backward();opt.step()
    gradients={n:p.grad.clone() for n,p in model.named_parameters()}
    path=tmp_path/'state.pt';torch.save(snapshot(model,opt,3),path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    other=build();other_opt,_=optimizer_for(other)
    other.load_state_dict(state['model']);other_opt.load_state_dict(state['optimizer'])
    equal_tree(model.state_dict(),other.state_dict());equal_tree(opt.state_dict(),other_opt.state_dict())
    restore_rng(state['rng']);expected=(torch.rand(8),torch.rand(8,device='cuda'))
    restore_rng(state['rng']);equal_tree(expected,(torch.rand(8),torch.rand(8,device='cuda')))
    for m,o in ((model,opt),(other,other_opt)):
        m.set_innovation_step(3);schedule(o,3,305176,wd)
        for n,p in m.named_parameters():p.grad=gradients[n].clone()
        o.step()
    equal_tree(model.state_dict(),other.state_dict());equal_tree(opt.state_dict(),other_opt.state_dict())
