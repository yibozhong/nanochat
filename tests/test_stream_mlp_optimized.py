"""Forward/all VJPs, boundary cases and full-model kernel substitution."""
from dataclasses import asdict
import json
from pathlib import Path
import pytest
import torch
from nanochat.stream_mlp import stream_mlp
from nanochat.stream_mlp_optimized import optimized_stream_mlp
from nanochat.stream_mlp_connections import StreamMLPHyperConnection
from nanochat.connection_model import ConnectionConfig,ConnectionModel
from nanochat.connection_experiment import configure_attention,match_vanilla_initialization
from nanochat.packed_training import seed_all
from test_stream_mlp import reference,relative

KINDS=('rowwise','hybrid')


def record(name,value):
    p=Path('results/stream-mlp-opt/checks');p.mkdir(parents=True,exist_ok=True)
    (p/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('n,c,amp',[(1,17,1.),(131,128,1.),(129,512,1.),(17,128,0.),(17,128,1e-4)])
def test_oracle(kind,n,c,amp):
    seed_all(209)
    args=[(torch.randn(1,n,c,4,device='cuda',dtype=torch.bfloat16)*amp).requires_grad_(),
          torch.randn(1,n,c,device='cuda',dtype=torch.bfloat16,requires_grad=True),
          torch.randn(1,n,4,device='cuda',requires_grad=True),
          (torch.randn(16,4,device='cuda')*.5).requires_grad_(),
          (torch.randn(4,16,device='cuda')*.25).requires_grad_()]
    output=optimized_stream_mlp(*args,kind);probe=torch.randn_like(output)
    grads=torch.autograd.grad(output,args,probe)
    refargs=[a.detach().double().requires_grad_() for a in args]
    ref=reference(*refargs);rg=torch.autograd.grad(ref,refargs,probe.double())
    errors=dict(forward=relative(output,ref),gradients=[relative(g,r) for g,r in zip(grads,rg)])
    record(f'{kind}-{n}-{c}-{amp}',errors)
    assert errors['forward']<.006 and max(errors['gradients'])<.012,errors
    again=torch.autograd.grad(optimized_stream_mlp(*args,kind),args,probe)
    assert all(torch.equal(a,b) for a,b in zip(grads,again))


@pytest.mark.parametrize('kind',KINDS)
def test_identity_and_saved_state(kind):
    seed_all(13);m=StreamMLPHyperConnection(128,4,0,stream_mlp_kernel=kind).cuda();m.reset_parameters()
    x=torch.randn(2,17,128,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.zeros_like(x[...,0]).contiguous();p=torch.zeros(2,17,4,device='cuda');saved=[]
    def pack(t):saved.append(tuple(t.shape));return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):out=m.write(y,x,p)
    assert relative(out,x)<1e-5 and len(saved)==5,saved
    changed=x.detach().clone();changed[0,3,5]+=1;after=m.write(y,changed,p)
    mask=torch.ones_like(out,dtype=torch.bool);mask[0,3,5]=False
    assert torch.equal(out[mask],after[mask])
    with torch.no_grad():m.mix_out.zero_()
    assert torch.count_nonzero(m.write(y,x,p))==0


@pytest.mark.parametrize('kind',KINDS)
def test_full_model(kind):
    configure_attention('hardware_auto');models=[]
    for impl in ('legacy',kind):
        seed_all(42)
        m=ConnectionModel(ConnectionConfig(variant='mhc_stream_mlp',width=128,heads=2,depth=2,
            vocab_size=128,compact_routing=True,stream_mlp_kernel=impl)).prepare(compile_blocks=True)
        match_vanilla_initialization(m,42)
        with torch.no_grad():
            for b in m.blocks:
                b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
                for c in (b.attn_connection,b.mlp_connection):
                    c.mix_in.normal_(std=.5);c.mix_out.normal_(std=.25)
        models.append(m)
    old,new=models
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    ids=torch.randint(0,128,(2,2048),device='cuda');changed=ids.clone();changed[0,1024:]=(changed[0,1024:]+13)%128
    with torch.no_grad():
        a,b=new(ids),new(changed)
        assert torch.equal(a[0,:1024],b[0,:1024]) and torch.equal(a[1],b[1])
    x=ids[:,:128].contiguous();y=ids[:,1:129].contiguous()
    losses=[m(x,y) for m in models]
    for loss in losses:loss.backward()
    error={k:relative(p.grad,dict(old.named_parameters())[k].grad) for k,p in new.named_parameters()}
    torch.testing.assert_close(*losses,atol=2e-4,rtol=2e-4)
    assert max(error.values())<.03,error
    grads={k:p.grad.clone() for k,p in new.named_parameters()}
    new.zero_grad(set_to_none=True);new.config.activation_checkpoint=True
    again=new(x,y);again.backward()
    rec={k:relative(p.grad,grads[k]) for k,p in new.named_parameters()}
    assert max(rec.values())<.02,rec
    record('model-'+kind,dict(ce=[l.item() for l in losses],gradients=error,recompute=rec))


def test_old_configuration_default():
    assert ConnectionConfig.from_checkpoint(dict(variant='mhc_stream_mlp')).stream_mlp_kernel=='legacy'
    with pytest.raises(ValueError):
        ConnectionConfig(variant='mhc_stream_mlp',stream_mlp_kernel='tensorcore')
    for kind in KINDS:
        c=ConnectionConfig(variant='mhc_stream_mlp',stream_mlp_kernel=kind)
        assert ConnectionConfig.from_checkpoint(asdict(c))==c
        with pytest.raises(ValueError):ConnectionConfig(stream_mlp_kernel=kind)
