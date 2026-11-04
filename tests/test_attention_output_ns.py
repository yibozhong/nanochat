"""Post-step W_o projection: independent spectral oracle and training integration."""
import copy
from dataclasses import asdict
import pytest
import torch
from nanochat.attention_output_ns import scaled_ns_projection, project_attention_outputs_, spectrum_summary, attention_output_diagnostics
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for,schedule
from nanochat.packed_training import seed_all,snapshot,restore_rng,train_step
from test_connection_training_resume import equal_tree


def build(enabled=True,compiled=False,length=128):
    seed_all(42);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.recompile_limit=64
    return ConnectionModel(ConnectionConfig(attention_output_ns=enabled,attention_kernel='fla_triton',
        depth=2,width=128,heads=1,vocab_size=256,sequence_length=length)).prepare(compile_blocks=compiled)


def test_fp64_scalar_singular_value_oracle():
    torch.manual_seed(17)
    u,_=torch.linalg.qr(torch.randn(16,16,device='cuda',dtype=torch.float64))
    v,_=torch.linalg.qr(torch.randn(16,16,device='cuda',dtype=torch.float64))
    s=torch.linspace(.1,2.,16,device='cuda',dtype=torch.float64)
    w=(u*s)@v.T
    # Independent scalar spectral map, with pinned coefficients from native Muon.
    coeffs=((8.156554524902461,-22.48329292557795,15.878769915207462),
        (4.042929935166739,-2.808917465908714,.5000178451051316),
        (3.8916678022926607,-2.772484153217685,.5060648178503393),
        (3.285753657755655,-2.3681294933425376,.46449024233003106),
        (2.3465413258596377,-1.7097828382687081,.42323551169305323))
    z=s/(s.norm()*1.01+1e-6)
    for a,b,c in coeffs:z=a*z+b*z**3+c*z**5
    expected=(u*(z*s.norm()/z.norm()))@v.T
    torch.testing.assert_close(scaled_ns_projection(w,torch.float64),expected,rtol=1e-11,atol=1e-11)


def test_real_512_bf16_spectrum_flattens_and_preserves_norm():
    torch.manual_seed(5)
    w=torch.randn(2,512,512,device='cuda')
    w[0]=torch.diag(torch.linspace(.1,2.,512,device='cuda'))
    result=scaled_ns_projection(w)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.norm(dim=(-2,-1)),w.norm(dim=(-2,-1)),rtol=2e-6,atol=1e-6)
    for before,after in zip(w,result):
        a=spectrum_summary(torch.linalg.svdvals(before.double()));b=spectrum_summary(torch.linalg.svdvals(after.double()))
        assert b['scaled_orthogonality_error']<.2
        assert b['scaled_orthogonality_error']<a['scaled_orthogonality_error']
        assert b['energy_erank']>a['energy_erank']


def test_zero_and_missing_directions_are_not_invented():
    w=torch.zeros(2,32,32,device='cuda');w[1,:8,:8]=torch.eye(8,device='cuda')
    q=scaled_ns_projection(w)
    assert torch.equal(q[0],w[0]) and q[1,8:].count_nonzero()==0 and q[1,:,8:].count_nonzero()==0
    assert spectrum_summary(torch.linalg.svdvals(q[1].double()))['numerical_rank']==8


def test_repeated_projection_is_finite_norm_preserving():
    torch.manual_seed(6)
    p=torch.nn.Parameter(torch.randn(2,128,128,device='cuda'))
    norms=p.norm(dim=(-2,-1)).detach().clone()
    for _ in range(20):
        with torch.no_grad():p.copy_(scaled_ns_projection(p))
    assert torch.isfinite(p).all() and p.grad is None
    torch.testing.assert_close(p.norm(dim=(-2,-1)),norms,rtol=5e-6,atol=1e-6)


def test_disabled_initialization_groups_and_rng_are_identical():
    base=build(False);rng=torch.get_rng_state().clone();ob,wd=optimizer_for(base)
    model=build();opt,newwd=optimizer_for(model)
    equal_tree(base.state_dict(),model.state_dict());assert torch.equal(rng,torch.get_rng_state())
    equal_tree(ob.state_dict(),opt.state_dict());assert wd==newwd
    assert not hasattr(ob,'_attention_output_ns_handle') and hasattr(opt,'_attention_output_ns_handle')
    x=torch.randint(256,(2,128),device='cuda')
    torch.testing.assert_close(base(x),model(x),rtol=0,atol=0)
    old=asdict(model.config);old.pop('attention_output_ns')
    assert not ConnectionConfig.from_checkpoint(old).attention_output_ns


@pytest.mark.parametrize('options',[dict(architecture='native'),dict(variant='mhc'),dict(attention_statistic='variance')])
def test_reject_uncontrolled_combinations(options):
    with pytest.raises(ValueError):ConnectionConfig(attention_output_ns=True,**options)


def test_only_output_weights_change_after_native_optimizer_step():
    base=build(False);model=build();ob,wd=optimizer_for(base);opt,_=optimizer_for(model)
    x=torch.randint(256,(2,128),device='cuda');y=(x+1)%256
    base(x,y).backward()
    grads={n:p.grad.clone() for n,p in base.named_parameters()}
    for n,p in model.named_parameters():p.grad=grads[n].clone()
    schedule(ob,0,305176,wd);schedule(opt,0,305176,wd);ob.step();opt.step()
    before=dict(base.named_parameters())
    for name,p in model.named_parameters():
        if name.endswith('attn.c_proj.weight'):
            expected=scaled_ns_projection(before[name])
            assert (p-expected).norm()/expected.norm()<.015
            torch.testing.assert_close(p.norm(),before[name].norm(),rtol=2e-6,atol=1e-7)
        else:assert torch.equal(p,before[name]),name
    # Projection is not an optimizer-state reset or a change to its momentum.
    equal_tree(ob.state_dict(),opt.state_dict())


def test_one_projection_after_complete_gradient_accumulation(monkeypatch):
    import nanochat.attention_output_ns as ns
    model=build();opt,wd=optimizer_for(model)
    x=torch.randint(256,(4,128),device='cuda');y=(x+1)%256
    class Data:
        def batch(self,split,start,n):return x[start:start+n],y[start:start+n]
    calls=[];original=ns.project_attention_outputs_
    def tracked(parameters):calls.append(len(parameters));original(parameters)
    monkeypatch.setattr(ns,'project_attention_outputs_',tracked)
    schedule(opt,0,305176,wd)
    loss=train_step(model,opt,Data(),0,microbatch=2,batch_tokens=512)
    assert calls==[2] and loss>0


def test_compiled_forward_observes_projected_weights_and_gradients():
    model=build(compiled=True);opt,wd=optimizer_for(model)
    x=torch.randint(256,(2,128),device='cuda');y=(x+1)%256
    first=model(x).detach()
    for step in range(3):
        opt.zero_grad(set_to_none=True);schedule(opt,step,305176,wd)
        loss=model(x,y);loss.backward();opt.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
    assert not torch.equal(first,model(x))


def test_diagnostics_preserve_model_rng_and_compiled_training():
    model=build(compiled=True);opt,wd=optimizer_for(model)
    x=torch.randint(256,(2,128),device='cuda');y=(x+1)%256
    model(x,y).backward();opt.step()
    class Data:
        arrays={'dev':range(2)}
        def batch(self,split,start,n):return x[start:start+n],y[start:start+n]
    before=copy.deepcopy(model.state_dict());optim=copy.deepcopy(opt.state_dict());compiled=model.compiled_steps
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
    report=attention_output_diagnostics(model,Data(),sequences=2,positions=128)
    equal_tree(before,model.state_dict());equal_tree(optim,opt.state_dict())
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
    assert model.training and model.compiled_steps is compiled and len(report['layers'])==2
    assert all(not b.attn.c_proj._forward_hooks for b in model.blocks)
    for row in report['layers']:
        assert row['weight']['energy_erank']>120
        assert row['input_features']['centered']['energy_erank']>0


def test_saved_optimizer_rng_and_fixed_gradient_next_step(tmp_path):
    model=build();opt,wd=optimizer_for(model)
    x=torch.randint(256,(2,128),device='cuda');y=(x+1)%256
    for step in range(3):
        opt.zero_grad(set_to_none=True);schedule(opt,step,305176,wd);model(x,y).backward();opt.step()
    gradients={n:p.grad.clone() for n,p in model.named_parameters()}
    path=tmp_path/'checkpoint.pt';torch.save(snapshot(model,opt,3,config=asdict(model.config)),path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    other=build();other_opt,_=optimizer_for(other)
    other.load_state_dict(state['model']);other_opt.load_state_dict(state['optimizer'])
    equal_tree(model.state_dict(),other.state_dict());equal_tree(opt.state_dict(),other_opt.state_dict())
    restore_rng(state['rng']);expected=(torch.rand(8),torch.rand(8,device='cuda'))
    restore_rng(state['rng']);equal_tree(expected,(torch.rand(8),torch.rand(8,device='cuda')))
    for m,o in ((model,opt),(other,other_opt)):
        for n,p in m.named_parameters():p.grad=gradients[n].clone()
        schedule(o,3,305176,wd);o.step()
    equal_tree(model.state_dict(),other.state_dict());equal_tree(opt.state_dict(),other_opt.state_dict())
