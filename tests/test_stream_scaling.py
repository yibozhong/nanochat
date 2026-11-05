"""Bounded scaling-benchmark correctness; no parameter optimization."""
import json
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from scripts.stream_scaling_ops import Aggregate,Write,MLP,Sinkhorn
from scripts.stream_scaling_model import GeneralConnection,ScalingModel
from nanochat.nvidia_connections import NvidiaHyperConnection
from nanochat.stream_mlp_connections import StreamMLPHyperConnection
from nanochat.connection_experiment import configure_attention


def relative(a,b):return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-12)).item()


def record(name,value):
    p=Path('results/stream-scaling/checks');p.mkdir(parents=True,exist_ok=True)
    (p/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


def ref_sinkhorn(x):
    f=torch.zeros_like(x[...,0]);g=torch.zeros_like(f)
    for _ in range(20):
        f=-torch.logsumexp(x+g.unsqueeze(-2),-1)
        g=-torch.logsumexp(x+f.unsqueeze(-1),-2)
    return (x+f.unsqueeze(-1)+g.unsqueeze(-2)).exp()


@pytest.mark.parametrize('n',(4,8,16,32))
def test_sinkhorn(n):
    torch.manual_seed(7);x=(torch.randn(37,n,n,device='cuda')*1.3).requires_grad_();go=torch.randn_like(x)
    actual=Sinkhorn.apply(x);gx,=torch.autograd.grad(actual,x,go)
    ref=x.detach().double().requires_grad_();expected=ref_sinkhorn(ref);eg,=torch.autograd.grad(expected,ref,go.double())
    errors=dict(output=relative(actual,expected),gradient=relative(gx,eg))
    record(f'sinkhorn-{n}',errors);assert max(errors.values())<2e-5,errors


@pytest.mark.parametrize('kind',('mhc','mlp'))
@pytest.mark.parametrize('n',(4,8,16,32))
def test_primitives(n,kind):
    torch.manual_seed(19);m,c=3,17
    x=torch.randn(m,c,n,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.randn(m,c,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    p=torch.randn(m,n,device='cuda',requires_grad=True)
    if kind=='mhc':
        a=torch.randn(m,n,n,device='cuda',requires_grad=True);args=(x,y,p,a)
        actual=Write.apply(*args)
        ref=lambda x,y,p,a:x@a+y.unsqueeze(-1)*p.unsqueeze(-2)
    else:
        wi=(torch.randn(4*n,n,device='cuda')/n**.5).requires_grad_()
        wo=(torch.randn(n,4*n,device='cuda')/(4*n)**.5).requires_grad_();args=(x,y,p,wi,wo)
        actual=MLP.apply(*args)
        ref=lambda x,y,p,wi,wo:F.linear(F.linear(x,wi).relu(),wo)+y.unsqueeze(-1)*p.unsqueeze(-2)
    probe=torch.randn_like(actual);grad=torch.autograd.grad(actual,args,probe)
    ra=[a.detach().double().requires_grad_() for a in args];expected=ref(*ra)
    rg=torch.autograd.grad(expected,ra,probe.double())
    errors=dict(output=relative(actual,expected),grads=[relative(g,r) for g,r in zip(grad,rg)])
    ag=Aggregate.apply(x,p);agp=torch.randn_like(ag);gg=torch.autograd.grad(ag,(x,p),agp)
    xx=x.detach().double().requires_grad_();pp=p.detach().double().requires_grad_()
    ar=(xx*pp[:,None,:]).sum(-1);gr=torch.autograd.grad(ar,(xx,pp),agp.double())
    errors['aggregate']=[relative(ag,ar),*[relative(g,r) for g,r in zip(gg,gr)]]
    record(f'primitives-{kind}-{n}',errors)
    assert errors['output']<.006 and max(errors['grads']+errors['aggregate'])<.012,errors


@pytest.mark.parametrize('kind',('mhc','mlp'))
@pytest.mark.parametrize('n',(4,8,16,32))
def test_connector(n,kind):
    torch.manual_seed(23);torch.backends.cuda.matmul.allow_tf32=True
    m=GeneralConnection(128,n,kind).cuda()
    if kind=='mlp':
        with torch.no_grad():m.win.normal_(std=n**-.5);m.wout.normal_(std=(4*n)**-.5)
    x=torch.randn(1,7,128,n,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.randn(1,7,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    branch,skip,state=m.read(x);out=m.write(y,skip,state)
    gb=torch.randn_like(branch);go=torch.randn_like(out);params=tuple(m.parameters())
    got=torch.autograd.grad((branch,out),(x,y,*params),(gb,go))
    xx=x.detach().double().requires_grad_();yy=y.detach().double().requires_grad_()
    rp={k:v.detach().double().requires_grad_() for k,v in m.named_parameters()}
    flat=xx.reshape(7,-1);raw=F.linear(flat,rp['phi']);r=(flat.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).rsqrt()
    pre=torch.sigmoid(raw[:,:n]*rp['scales'][0]*r+rp['bias'][:n])
    post=2*torch.sigmoid(raw[:,n:2*n]*rp['scales'][1]*r+rp['bias'][n:2*n])
    refbranch=(xx*pre.reshape(1,7,1,n)).sum(-1)
    if kind=='mhc':
        coef=ref_sinkhorn((raw[:,2*n:]*rp['scales'][2]*r+rp['bias'][2*n:]).reshape(1,7,n,n))
        refout=xx@coef+yy.unsqueeze(-1)*post.reshape(1,7,1,n)
    else:refout=F.linear(F.linear(xx,rp['win']).relu(),rp['wout'])+yy.unsqueeze(-1)*post.reshape(1,7,1,n)
    expected=torch.autograd.grad((refbranch,refout),(xx,yy,*rp.values()),(gb.double(),go.double()))
    errors=dict(branch=relative(branch,refbranch),output=relative(out,refout),grads=[relative(g,e) for g,e in zip(got,expected)])
    record(f'connector-{kind}-{n}',errors)
    assert max(errors['branch'],errors['output'])<.006 and max(errors['grads'])<.02,errors


@pytest.mark.parametrize('kind',('mhc','mlp'))
def test_existing_four_stream_reference(kind):
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=True
    m=GeneralConnection(128,4,kind).cuda()
    if kind=='mlp':
        with torch.no_grad():m.win.normal_(std=.5);m.wout.normal_(std=.25)
    old=(NvidiaHyperConnection(128,4,0) if kind=='mhc' else StreamMLPHyperConnection(128,4,0)).cuda()
    old.reset_parameters()
    with torch.no_grad():
        old.phi.copy_(m.phi);old.bias.copy_(m.bias);old.scales.copy_(m.scales[:old.scales.numel()])
        if kind=='mlp':old.mix_in.copy_(m.win);old.mix_out.copy_(m.wout)
    x=torch.randn(2,19,128,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.randn(2,19,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    gb=torch.randn_like(y);go=torch.randn_like(x);values=[];grads=[]
    for model in (m,old):
        b,s,state=model.read(x);o=model.write(y,s,state);values.append((b,o))
        grads.append(torch.autograd.grad((b,o),(x,y),(gb,go)))
    errors=[relative(a,b) for a,b in zip(values[0]+grads[0],values[1]+grads[1])]
    record('existing4-'+kind,errors);assert max(errors)<.02,errors


@pytest.mark.parametrize('kind',('mhc','mlp'))
@pytest.mark.parametrize('n',(4,8,16,32))
def test_full_model_causality_and_no_updates(n,kind):
    configure_attention('hardware_auto');torch.backends.cuda.matmul.allow_tf32=True
    model=ScalingModel(kind,n,width=128,depth=2,vocab=128)
    ids=torch.randint(0,128,(1,2048),device='cuda');changed=ids.clone();changed[:,1024:]=(changed[:,1024:]+11)%128
    with torch.no_grad():
        a=model.hidden(ids);b=model.hidden(changed)
        assert torch.equal(a[:,:1024],b[:,:1024])
    params=tuple(model.parameters());versions=[p._version for p in params]
    loss=model(ids[:,:128].contiguous(),ids[:,1:129].contiguous())
    grads=torch.autograd.grad(loss,params)
    assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads)
    assert versions==[p._version for p in params]
    record(f'full-{kind}-{n}',dict(loss=loss.item(),no_updates=True,causal_tokens=2048))
