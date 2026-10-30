"""Independent math and model integration checks; run on a compute node."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from nanochat.hyper_connections import HyperConnection, ManifoldHyperConnection, ManifoldHyperHead, sinkhorn
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for
from nanochat.engine import KVCache


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


@pytest.mark.parametrize('kind',['hc','mhc'])
def test_independent_forward_and_parameter_gradients(kind):
    torch.manual_seed(47)
    cls=HyperConnection if kind=='hc' else ManifoldHyperConnection
    m=cls(6,4,1).double();m.reset_parameters()
    with torch.no_grad():
        for p in m.parameters():p.add_(.07*torch.randn_like(p))
    x=torch.randn(2,3,4,6,dtype=torch.float64,requires_grad=True)
    y=torch.randn(2,3,6,dtype=torch.float64,requires_grad=True)
    branch,skip,post=m.read(x);actual=m.write(y,skip,post)
    if kind=='hc':
        z=F.layer_norm(x,(6,),m.route_norm.weight,m.route_norm.bias,1e-5)
        alpha=m.alpha+m.scale_alpha*torch.tanh(torch.einsum('btid,dk->btik',z,m.phi_alpha))
        beta=m.beta+m.scale_beta*torch.tanh(torch.einsum('btid,d->bti',z,m.phi_beta))
        pre=alpha[...,0];comb=alpha[...,1:]
    else:
        # Official DeepSeek ordering: fp32/64 projection, then RMS factor, gates,
        # and source->destination combination. Use the unmodified official SK oracle.
        ref=load(Path(__file__).parents[1]/'research/connections/references/official_sinkhorn.py','official_sk')
        flat=x.flatten(-2)
        raw=F.linear(flat,m.phi)*torch.rsqrt(flat.square().mean(-1,keepdim=True)+1e-6)
        pre=torch.sigmoid(raw[...,:4]*m.scales[0]+m.bias[:4])+1e-6
        beta=2*torch.sigmoid(raw[...,4:8]*m.scales[1]+m.bias[4:8])
        comb=ref._sinkhorn_reference((raw[...,8:]*m.scales[2]+m.bias[8:]).reshape(2,3,4,4),20,1e-6)
    ref_branch=sum(pre[...,i,None]*x[...,i,:] for i in range(4))
    expected=torch.stack([sum(comb[...,i,j,None]*x[...,i,:] for i in range(4))+beta[...,j,None]*y
                          for j in range(4)],dim=-2)
    torch.testing.assert_close(branch,ref_branch,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    probe=torch.randn_like(actual);q=torch.randn_like(branch)
    inputs=[x,y,*m.parameters()]
    g1=torch.autograd.grad((actual*probe).sum()+(branch*q).sum(),inputs,retain_graph=True)
    g2=torch.autograd.grad((expected*probe).sum()+(ref_branch*q).sum(),inputs)
    for a,b in zip(g1,g2):torch.testing.assert_close(a,b,atol=2e-10,rtol=2e-10)


def test_sinkhorn_gradcheck_and_constraints():
    torch.manual_seed(19)
    x=(.5*torch.randn(2,4,4,dtype=torch.float64)).requires_grad_()
    assert torch.autograd.gradcheck(sinkhorn,(x,),atol=1e-5,rtol=1e-4)
    p=sinkhorn(x)
    assert p.min()>0
    for dim in (-1,-2):assert (p.sum(dim)-1).abs().max()<1e-5
    # Extreme logits remain finite, but finite Sinkhorn is not claimed exactly DS.
    assert torch.isfinite(sinkhorn(x*10000)).all()


def test_hc_initialization_and_identity_branch():
    m=HyperConnection(12,4,3).double()
    h=torch.randn(2,5,12,dtype=torch.float64)
    x=h.unsqueeze(-2).expand(-1,-1,4,-1)
    branch,skip,post=m.read(x)
    torch.testing.assert_close(branch,h,atol=0,rtol=0)
    torch.testing.assert_close(skip,x,atol=0,rtol=0)
    y=torch.randn_like(h)
    torch.testing.assert_close(m.write(y,skip,post),(h+y).unsqueeze(-2).expand_as(x),atol=0,rtol=0)


def test_mhc_head_official_formula():
    m=ManifoldHyperHead(6,4).double();m.reset_parameters()
    x=torch.randn(2,3,4,6,dtype=torch.float64,requires_grad=True)
    flat=x.flatten(2)
    mixes=F.linear(flat,m.phi)*torch.rsqrt(flat.square().mean(-1,keepdim=True)+1e-6)
    pre=torch.sigmoid(mixes*m.scale+m.bias)+1e-6
    expected=torch.sum(pre.unsqueeze(-1)*x,dim=2)
    torch.testing.assert_close(m(x),expected,atol=0,rtol=0)


def test_backbone_seeds_and_optimizer_coverage():
    previous=None
    for variant in ('baseline','hc','mhc'):
        torch.manual_seed(42)
        m=ConnectionModel(ConnectionConfig(variant=variant,width=32,heads=2,depth=2,vocab_size=64,sequence_length=16))
        backbone={k:p.detach().clone() for k,p in m.named_parameters() if '_connection.' not in k and not k.startswith('stream_head.')}
        if previous:
            for k,p in backbone.items():torch.testing.assert_close(p,previous[k],atol=0,rtol=0)
        previous=backbone
        opt,_=optimizer_for(m)
        ids=[id(p) for g in opt.param_groups for p in g['params']]
        assert len(ids)==len(set(ids))==len(list(m.parameters()))
        routing={id(p) for n,p in m.named_parameters() if '_connection.' in n or n.startswith('stream_head.')}
        assert all(g['kind']=='adamw' for g in opt.param_groups if any(id(p) in routing for p in g['params']))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU integration')
@pytest.mark.parametrize('variant',['baseline','hc','mhc'])
def test_gpu_causality_cache_cce_and_gradients(variant):
    torch.manual_seed(33)
    c=ConnectionConfig(variant=variant,width=64,heads=2,depth=2,vocab_size=128,sequence_length=2048)
    m=ConnectionModel(c).prepare()
    with torch.no_grad():
        for b in m.blocks:
            b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
    ids=torch.randint(0,128,(1,2048),device='cuda')
    changed=ids.clone();changed[:,1024:]=(changed[:,1024:]+17)%128
    with torch.no_grad():
        a=m(ids);b=m(changed)
        torch.testing.assert_close(a[:,:1024],b[:,:1024],atol=0,rtol=0)
        cache=KVCache(1,2,32,32,2,'cuda',torch.bfloat16)
        pre=m(ids[:,:7],kv_cache=cache)
        next_token=m(ids[:,7:8],kv_cache=cache)
        full=m(ids[:,:8])
        torch.testing.assert_close(pre,full[:,:7],atol=.004,rtol=.03)
        torch.testing.assert_close(next_token,full[:,7:],atol=.004,rtol=.03)
    x,y=ids[:,:16].contiguous(),ids[:,1:17].contiguous()
    loss=m(x,y);dense=m(x,y,dense=True)
    torch.testing.assert_close(loss,dense,atol=2e-4,rtol=2e-4)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    if variant!='baseline':
        assert sum(p.grad.float().square().sum().item() for n,p in m.named_parameters() if '_connection.' in n)>0
    a=copy.deepcopy(m.state_dict());n=ConnectionModel(c).prepare();n.load_state_dict(a)
    for key,tensor in n.state_dict().items():torch.testing.assert_close(tensor,a[key],atol=0,rtol=0)


@pytest.mark.skipif(not os.environ.get('LEGACY_BASELINE_SOURCE'),reason='Archived baseline supplied by validation job')
def test_exact_archived_baseline_equivalence():
    old=load(os.environ['LEGACY_BASELINE_SOURCE'],'archived_full_attention')
    opts=dict(depth=2,width=32,heads=2,sequence_length=16,vocab_size=64,activation_checkpoint=False)
    torch.manual_seed(31);a=old.Model(old.Config(**opts))
    torch.manual_seed(31);b=ConnectionModel(ConnectionConfig(**opts))
    assert set(a.state_dict())==set(b.state_dict())
    for name,p in a.state_dict().items():torch.testing.assert_close(p,b.state_dict()[name],atol=0,rtol=0)
    ids=torch.randint(0,64,(2,16))
    with patch('nanochat.flash_attention.USE_FA3',False):
        torch.testing.assert_close(a.hidden(ids),b.hidden(ids),atol=0,rtol=0)
    oa,wa=old.optimizer_for(a);ob,wb=optimizer_for(b)
    assert wa==wb
    for ga,gb in zip(oa.param_groups,ob.param_groups):
        assert {k:v for k,v in ga.items() if k!='params'}=={k:v for k,v in gb.items() if k!='params'}
