"""Official mHC kernel forward/backward and nanochat integration checks."""
import copy
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for
from nanochat.manifold_connections import ManifoldHyperConnection, ManifoldHyperHead, sinkhorn_normalize
from nanochat.engine import KVCache


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


def relative_error(actual, expected):
    delta=(actual.float()-expected.float()).norm()
    return (delta/(expected.float().norm()+1e-12)).item()



def test_installed_official_sources_are_unmodified():
    import tile_kernels
    manifest=json.loads((Path(__file__).parents[1]/'research/connections/official-files.json').read_text())
    root=Path(tile_kernels.__file__).parent
    for name,digest in manifest['sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name


def reference_read(m, x):
    from tile_kernels.torch.mhc import mhc_pre_norm_fn_ref, mhc_pre_split_mixes_ref, sinkhorn_normalize_ref, mhc_pre_apply_mix_ref
    mixes=mhc_pre_norm_fn_ref(x,m.phi,None,1e-6)
    pre,post,comb=mhc_pre_split_mixes_ref(mixes,m.scales,m.bias,4,2.,1e-6)
    if m.manifold:
        comb=sinkhorn_normalize_ref(comb,repeat=20,eps=1e-6)
    return mhc_pre_apply_mix_ref(x,pre),(post,comb)


@pytest.mark.parametrize('width',[128,512])
@pytest.mark.parametrize('manifold',[True,False])
def test_official_pipeline_against_official_torch_reference(width,manifold):
    from tile_kernels.torch.mhc import mhc_post_ref
    torch.manual_seed(47)
    m=ManifoldHyperConnection(width,4,0,manifold=manifold).cuda();m.reset_parameters()
    n=copy.deepcopy(m)
    x=torch.randn(1,256,4,width,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.randn(1,256,width,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    xr=x.detach().clone().requires_grad_();yr=y.detach().clone().requires_grad_()
    branch,residual,mixes=m.read(x);actual=m.write(y,residual,mixes)
    if not manifold:
        assert mixes[1].min()<0  # Official post supports signed, unnormalized mixing.
        assert (mixes[1].sum(-1)-1).abs().max()>.001
    old=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=True
    try:
        ref_branch,ref_mixes=reference_read(n,xr)
        expected=mhc_post_ref(yr,xr,*ref_mixes)
        p=torch.randn_like(actual);q=torch.randn_like(branch)
        g1=torch.autograd.grad((actual*p).float().mean()+(branch*q).float().mean(),[x,y,*m.parameters()])
        g2=torch.autograd.grad((expected*p).float().mean()+(ref_branch*q).float().mean(),[xr,yr,*n.parameters()])
    finally:torch.backends.cuda.matmul.allow_tf32=old
    errors=dict(output=relative_error(actual,expected),branch=relative_error(branch,ref_branch),
                gradients=[relative_error(a,b) for a,b in zip(g1,g2)])
    # BF16 outputs/activation gradients and official TF32 projection rounding.
    assert errors['output']<.01 and errors['branch']<.01,errors
    assert max(errors['gradients'])<.03,errors
    assert all(torch.isfinite(a).all() for a in g1)
    with torch.no_grad():
        fused,_,ctx=m.read(x)
        assert relative_error(fused,branch)<.01
        assert relative_error(ctx[0],mixes[0])<.01
        assert relative_error(ctx[1],mixes[1])<.01
    Path('results').mkdir(exist_ok=True)
    suffix='' if manifold else '-no-manifold'
    Path(f'results/kernel-errors-d{width}{suffix}.json').write_text(json.dumps(errors,indent=2))


def test_no_manifold_never_calls_sinkhorn_in_training_or_inference():
    from nanochat.manifold_connections import official_ops
    m=ManifoldHyperConnection(128,4,0,manifold=False).cuda();m.reset_parameters()
    x=torch.randn(1,32,4,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    with patch('nanochat.manifold_connections.sinkhorn_normalize',side_effect=AssertionError('Sinkhorn called')), \
         patch.object(official_ops(),'mhc_pre_big_fuse',side_effect=AssertionError('Fused Sinkhorn called')):
        branch,residual,mixes=m.read(x)
        output=m.write(branch,residual,mixes)
        output.float().square().mean().backward()
        with torch.no_grad():
            branch_eval,residual_eval,mixes_eval=m.read(x)
            output_eval=m.write(branch_eval,residual_eval,mixes_eval)
        torch.testing.assert_close(output,output_eval,atol=0,rtol=0)


def test_official_head_gradients():
    from tile_kernels.torch.mhc import mhc_pre_norm_fn_ref,mhc_head_compute_mix_ref,mhc_pre_apply_mix_ref
    torch.manual_seed(23)
    m=ManifoldHyperHead(512,4).cuda();m.reset_parameters();n=copy.deepcopy(m)
    x=torch.randn(1,256,4,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    xr=x.detach().clone().requires_grad_()
    actual=m(x)
    old=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=True
    try:
        raw=mhc_pre_norm_fn_ref(xr,n.phi,None,1e-6)
        pre=mhc_head_compute_mix_ref(raw,n.scale,n.bias,1e-6)
        expected=mhc_pre_apply_mix_ref(xr,pre.unsqueeze(-1))
        probe=torch.randn_like(actual)
        ga=torch.autograd.grad((actual*probe).float().mean(),[x,*m.parameters()])
        gb=torch.autograd.grad((expected*probe).float().mean(),[xr,*n.parameters()])
    finally:torch.backends.cuda.matmul.allow_tf32=old
    assert relative_error(actual,expected)<.01
    assert max(relative_error(a,b) for a,b in zip(ga,gb))<.03


def test_official_sinkhorn_constraints_and_gradients():
    from tile_kernels.torch.mhc import sinkhorn_normalize_ref
    x=torch.randn(1,256,4,4,device='cuda',requires_grad=True)
    xr=x.detach().clone().requires_grad_()
    actual=sinkhorn_normalize(x,repeat=20,eps=1e-6)
    expected=sinkhorn_normalize_ref(xr,repeat=20,eps=1e-6)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    assert actual.min()>0
    for dim in (-1,-2):assert (actual.sum(dim)-1).abs().max()<1e-5
    probe=torch.randn_like(actual)
    ga=torch.autograd.grad((actual*probe).sum(),x)[0]
    gb=torch.autograd.grad((expected*probe).sum(),xr)[0]
    assert relative_error(ga,gb)<1e-4


def test_backbone_seeds_and_optimizer_coverage():
    previous=None
    connection_state=None
    for variant in ('baseline','mhc','mhc_no_manifold'):
        torch.manual_seed(42)
        m=ConnectionModel(ConnectionConfig(variant=variant,width=128,heads=2,depth=2,vocab_size=64,sequence_length=16))
        backbone={k:p.detach().clone() for k,p in m.named_parameters() if '_connection.' not in k and not k.startswith('stream_head.')}
        if previous:
            for k,p in backbone.items():torch.testing.assert_close(p,previous[k],atol=0,rtol=0)
        previous=backbone
        if variant=='mhc':connection_state=copy.deepcopy(m.state_dict())
        if variant=='mhc_no_manifold':
            for key,tensor in m.state_dict().items():
                torch.testing.assert_close(tensor,connection_state[key],atol=0,rtol=0)
        opt,_=optimizer_for(m)
        ids=[id(p) for g in opt.param_groups for p in g['params']]
        assert len(ids)==len(set(ids))==len(list(m.parameters()))
        routing={id(p) for n,p in m.named_parameters() if '_connection.' in n or n.startswith('stream_head.')}
        assert all(g['kind']=='adamw' for g in opt.param_groups if any(id(p) in routing for p in g['params']))
    with pytest.raises(ValueError):ConnectionConfig(variant='hc')
    with pytest.raises(ValueError):ConnectionConfig(variant='mhc',streams=2)


@pytest.mark.parametrize('variant',['baseline','mhc','mhc_no_manifold'])
def test_gpu_causality_cache_cce_and_gradients(variant):
    torch.manual_seed(33)
    c=ConnectionConfig(variant=variant,width=128,heads=2,depth=2,vocab_size=128,sequence_length=2048)
    m=ConnectionModel(c).prepare()
    with torch.no_grad():
        for b in m.blocks:
            b.attn.c_proj.weight.normal_(std=.03);b.mlp.c_proj.weight.normal_(std=.03)
    ids=torch.randint(0,128,(1,2048),device='cuda')
    changed=ids.clone();changed[:,1024:]=(changed[:,1024:]+17)%128
    with torch.no_grad():
        a=m(ids);b=m(changed)
        torch.testing.assert_close(a[:,:1024],b[:,:1024],atol=0,rtol=0)
        cache=KVCache(1,2,32,64,2,'cuda',torch.bfloat16)
        pre=m(ids[:,:7],kv_cache=cache);next_token=m(ids[:,7:8],kv_cache=cache)
        full=m(ids[:,:8])
        torch.testing.assert_close(pre,full[:,:7],atol=.004,rtol=.03)
        torch.testing.assert_close(next_token,full[:,7:],atol=.004,rtol=.03)
    x,y=ids[:,:128].contiguous(),ids[:,1:129].contiguous()
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
