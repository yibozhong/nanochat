"""Read-only diagnosis of the stopped final-mixing checkpoint.

The same states support all frozen counterfactuals. No comparison between
different training stages, optimizer updates, or W&B runs is performed.
"""
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from nanochat.connection_model import ConnectionConfig, ConnectionModel
from nanochat.connection_experiment import configure_attention
from nanochat.packed_training import seed_all
from nanochat.gpt import norm
from scripts import connection_rank_diagnostics as spectra

OUT=Path('results')


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def relative(a,b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30)).item()


def contrast_basis(device):
    e=torch.ones(4,device=device,dtype=torch.float64)/2
    c=torch.tensor([[1,1,1],[-1,1,1],[0,-2,1],[0,0,-3]],device=device,dtype=torch.float64)
    return e,c/c.norm(dim=0)


def reference_matrix(x,phi,bias,scale):
    flat=x.flatten(-2)
    raw=F.linear(flat/(flat.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt(),phi)
    z=(raw*scale+bias).reshape(*x.shape[:-2],4,4)
    for _ in range(20):
        z=z-z.logsumexp(-1,keepdim=True)
        z=z-z.logsumexp(-2,keepdim=True)
    return z.exp()


def matrix_summary(matrix):
    m=matrix.double();e,c=contrast_basis(m.device)
    s=torch.linalg.svdvals(m);contrast=c.T@m@c;cs=torch.linalg.svdvals(contrast)
    def quantiles(x):
        return {str(q):torch.quantile(x.flatten(),q).item() for q in (0.,.01,.1,.5,.9,.99,1.)}
    orth=(m.transpose(-1,-2)@m-torch.eye(4,device=m.device)).square().sum((-1,-2))
    return dict(tokens=len(m),matrix_mean=m.mean(0).cpu().tolist(),
        singular_values=[quantiles(s[:,i]) for i in range(4)],
        contrast_singular_values=[quantiles(cs[:,i]) for i in range(3)],
        condition_number=quantiles(s[:,0]/s[:,-1].clamp_min(1e-30)),
        fraction_smallest_singular_below={str(q):(s[:,-1]<q).double().mean().item() for q in (1e-6,1e-4,.01,.1)},
        mean_squared_orthogonality_error=orth.mean().item(),
        row_sum_max_error=(m.sum(-1)-1).abs().max().item(),
        column_sum_max_error=(m.sum(-2)-1).abs().max().item(),
        coefficient_rank_note='Coefficient matrix only; input-dependent routing adds another Jacobian term')


def common_preserving_polar(m):
    e,c=contrast_basis(m.device)
    u,_,vh=torch.linalg.svd(c.T@m.double()@c)
    return e[:,None]*e[None,:]+c@(u@vh)@c.T


def summarize_state(h,g,name):
    h=h.cuda().float();g=g.cuda().float()
    # Stored stream-major [N,K,D].
    return dict(activation=dict(streams=spectra.stream_stats(h),
        centered=spectra.matrix_spectrum(h.flatten(1),name+'_activation_centered',True)),
        gradient=dict(streams=spectra.stream_stats(g),
        uncentered=spectra.matrix_spectrum(g.flatten(1),name+'_gradient'),
        centered=spectra.matrix_spectrum(g.flatten(1),name+'_gradient_centered',True)))


def sampled_ce(h,weight,ids,inverse,targets):
    values=[]
    for start in range(0,len(h),128):
        x=h[start:start+128].float()
        pre=torch.cat([x[:,i]@weight[part].T for i,part in enumerate(ids)],-1)
        logits=15*(pre/15).tanh()
        values.append(F.cross_entropy(logits,inverse[targets[start:start+128]],reduction='none').double().cpu())
    return torch.cat(values)


def self_test():
    # Check the direct/routing split with a small independent FP64 graph.
    torch.manual_seed(24)
    x=torch.randn(13,5,4,device='cuda',dtype=torch.float64,requires_grad=True)
    phi=torch.randn(16,20,device='cuda',dtype=torch.float64)*.3
    bias=torch.eye(4,device='cuda',dtype=torch.float64).flatten()
    scale=torch.tensor([.4],device='cuda',dtype=torch.float64)
    m=reference_matrix(x,phi,bias,scale)
    y=x@m;g=torch.randn_like(y)
    total=torch.autograd.grad(y,x,g,retain_graph=True)[0]
    direct=g@m.detach().transpose(-1,-2)
    gm=x.detach().transpose(-1,-2)@g
    route=torch.autograd.grad(m,x,gm)[0]
    torch.testing.assert_close(total,direct+route,atol=1e-11,rtol=1e-10)
    q=common_preserving_polar(m.detach());eye=torch.eye(4,device='cuda',dtype=torch.float64)
    torch.testing.assert_close(q.transpose(-1,-2)@q,eye.expand_as(q),atol=1e-11,rtol=1e-10)
    torch.testing.assert_close(q.sum(-1),torch.ones_like(q[...,0]),atol=1e-11,rtol=1e-10)
    common=x.detach().mean(-1,keepdim=True).expand_as(x)
    torch.testing.assert_close(common@q,common,atol=1e-11,rtol=1e-10)
    spectra.self_test()
    dump('mixing-self-test.json',dict(passed=True,gradient_decomposition=True,
         orthogonal_common_preservation=True,spectral_oracle=True))


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    configure_attention('sdpa_cudnn');seed_all(42);self_test()
    plan=json.loads(Path('../diagnostic-inputs.json').read_text());case=plan['cases'][0]
    ckpt=torch.load(case['path'],map_location='cpu',weights_only=False)
    assert ckpt['run_id']==case['run_id'] and ckpt['step']==case['expected_step']
    config=ConnectionConfig.from_checkpoint(ckpt['config'])
    assert config.readout=='vocab_partition' and config.partition_mixing=='mhc'
    model=ConnectionModel(config).prepare(compile_blocks=True)
    model.load_state_dict(ckpt['model'],strict=True);model.eval();del ckpt;gc.collect()
    manifest=json.loads(Path('../data/manifest.json').read_text())
    dev_path=Path('../data/dev.bin')
    assert hashlib.sha256(dev_path.read_bytes()).hexdigest()==manifest['files']['dev']['sha256']
    dev=np.memmap(dev_path,mode='r',dtype=np.uint16,shape=(64,2049))
    rng=np.random.default_rng(20261031)
    positions=np.stack([np.sort(rng.choice(2048,128,replace=False)) for _ in range(64)])
    capture={};hs={};gs={};targets=[];matrices=[];directs=[];routes=[];errs=[]
    # Retain original graph adjoints, including dynamic coefficients.
    for i,block in enumerate(model.blocks):
        original=block.forward
        def traced(*args,index=i,forward=original,**kwargs):
            y=forward(*args,**kwargs);y.retain_grad();capture[f'block{index+1}']=y
            return y
        block.forward=traced
    mixer=model.partition_mixer;original_matrix=mixer.mixing_matrix
    def traced_matrix(x):
        x.retain_grad();capture['before_mix']=x
        m=original_matrix(x);m.retain_grad();capture['matrix']=m
        return m
    mixer.mixing_matrix=traced_matrix
    def after_hook(module,args,y):y.retain_grad();capture['after_mix']=y
    hook=mixer.register_forward_hook(after_hook)
    original_hidden=model.hidden
    def traced_hidden(*args,**kwargs):
        h=original_hidden(*args,**kwargs);h.retain_grad();capture['head_normalized']=h
        return h
    model.hidden=traced_hidden
    total_ce=0.;start_time=time.time()
    for start in range(0,64,2):
        row=torch.tensor(np.array(dev[start:start+2],dtype=np.int64,copy=True),device='cuda')
        x,y=row[:,:-1].contiguous(),row[:,1:].contiguous();model.zero_grad(set_to_none=True);capture.clear()
        loss=model(x,y);loss.backward(retain_graph=True);total_ce+=loss.item()*y.numel()
        pos=torch.as_tensor(positions[start:start+2],device='cuda');batch=torch.arange(2,device='cuda')[:,None]
        targets.append(y[batch,pos].flatten().cpu())
        for name,z in capture.items():
            if name=='matrix':continue
            a=z.detach()[batch,pos];g=z.grad.detach()[batch,pos].float()*y.numel()
            if name!='head_normalized':a=a.transpose(-1,-2);g=g.transpose(-1,-2)
            hs.setdefault(name,[]).append(a.reshape(-1,4,config.width).cpu())
            gs.setdefault(name,[]).append(g.reshape(-1,4,config.width).cpu())
        mat=capture['matrix'];before=capture['before_mix'];after=capture['after_mix']
        actual_before=before.grad.detach().float().clone()
        direct=after.grad.detach().float()@mat.detach().transpose(-1,-2)
        route=torch.autograd.grad(mat,before,mat.grad.detach().clone())[0].detach().float()
        error=relative(direct+route,actual_before);errs.append(error)
        assert error<.03,error
        directs.append((direct[batch,pos]*y.numel()).transpose(-1,-2).reshape(-1,4,config.width).cpu())
        routes.append((route[batch,pos]*y.numel()).transpose(-1,-2).reshape(-1,4,config.width).cpu())
        matrices.append(mat.detach()[batch,pos].reshape(-1,4,4).cpu())
        print(json.dumps(dict(dev_rows=start+2,loss=loss.item(),vjp_decomposition_relative_error=error)),flush=True)
        capture.clear();del loss,mat,before,after,direct,route,actual_before
    hook.remove();mixer.mixing_matrix=original_matrix;model.hidden=original_hidden
    model.zero_grad(set_to_none=True)
    hs={k:torch.cat(v) for k,v in hs.items()};gs={k:torch.cat(v) for k,v in gs.items()}
    mats=torch.cat(matrices).cuda();target=torch.cat(targets).cuda()
    direct=torch.cat(directs).cuda().float();route=torch.cat(routes).cuda().float()
    actual=gs['before_mix'].cuda().float()
    geom=matrix_summary(mats)
    result=dict(complete=False,logging_backend='local',plan=plan,gpu=torch.cuda.get_device_name(),
        checkpoint_step=case['expected_step'],training_tokens=case['expected_step']*32768,
        dev_tokens=131072,sampled_tokens=8192,dev_ce=total_ce/131072,
        sample_positions_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),
        matrix=geom,sites={},vjp=dict(max_relative_reconstruction_error=max(errs),
        direct_norm_over_total=(direct.norm()/actual.norm()).item(),
        routing_norm_over_total=(route.norm()/actual.norm()).item(),
        direct_routing_cosine=(torch.sum(direct*route)/(direct.norm()*route.norm()).clamp_min(1e-30)).item(),
        definition='Actual upstream adjoint= fixed-coefficient residual VJP + input-dependent coefficient VJP, within BF16 accumulation tolerance'))
    dump('mixing-partial.json',result)
    for name,h in hs.items():
        result['sites'][name]=summarize_state(h,gs[name],name)
        print(json.dumps(dict(site=name,activation_erank=result['sites'][name]['activation']['centered']['energy_entropy_rank'],gradient_erank=result['sites'][name]['gradient']['uncentered']['energy_entropy_rank'])),flush=True)
    result['vjp']['direct_spectrum']=spectra.matrix_spectrum(direct.flatten(1),'direct_vjp')
    result['vjp']['routing_spectrum']=spectra.matrix_spectrum(route.flatten(1),'routing_vjp')
    result['vjp']['direct_streams']=spectra.stream_stats(direct)
    result['vjp']['routing_streams']=spectra.stream_stats(route)
    del direct,route,actual;gc.collect();torch.cuda.empty_cache()
    # Frozen probes preserve the head weights; only final mixing is changed.
    raw=hs['before_mix'].cuda().transpose(-1,-2).double()
    q=common_preserving_polar(mats)
    identity=torch.eye(4,device='cuda',dtype=torch.float64)
    weights=model.lm_head.weight.detach().bfloat16().float()
    ids,inverse=model.partition_ids,model.partition_inverse
    baseline=sampled_ce(hs['head_normalized'].cuda(),weights,ids,inverse,target)
    result['probes']={}
    alternatives={
        'original':hs['head_normalized'].cuda(),
        'reconstructed_original':norm((raw@mats.double()).transpose(-1,-2).contiguous().bfloat16()),
        'identity_bypass':norm(raw.transpose(-1,-2).contiguous().bfloat16()),
        'half_identity_half_original':norm((raw@(.5*identity+.5*mats.double())).transpose(-1,-2).contiguous().bfloat16()),
        'orthogonal_common_preserving':norm((raw@q).transpose(-1,-2).contiguous().bfloat16()),
        'mean_all_streams':norm(raw.mean(-1,keepdim=True).expand_as(raw).transpose(-1,-2).contiguous().bfloat16())}
    losses={}
    for name,h in alternatives.items():
        ce=baseline if name=='original' else sampled_ce(h,weights,ids,inverse,target)
        delta=ce-baseline;paired=delta.reshape(64,128).mean(1)
        losses[name]=ce.numpy()
        result['probes'][name]=dict(ce=ce.mean().item(),delta_ce=delta.mean().item(),
            delta_halves=[v.mean().item() for v in delta.chunk(2)],
            sequence_mean_delta_se=(paired.std(unbiased=True)/len(paired)**.5).item(),
            worse_sequence_fraction=(paired>0).double().mean().item(),
            normalized_hidden_relative_change=relative(h,alternatives['original']),
            normalized_streams=spectra.stream_stats(h))
        print(json.dumps(dict(probe=name,ce=ce.mean().item(),delta=delta.mean().item())),flush=True)
    result['orthogonal_probe']={
        'max_orthogonality_error':(q.transpose(-1,-2)@q-identity).abs().max().item(),
        'max_common_preservation_error':(q.sum(-1)-1).abs().max().item(),
        'negative_entry_fraction':(q<0).double().mean().item(),
        'note':'Polar factor only in the3D stream-contrast subspace; the common direction remains fixed. Signed mixing, not nonnegative Sinkhorn.'}
    result['activation_disagreement_energy_retention']=float(
        ((hs['after_mix'].double()-hs['after_mix'].double().mean(1,keepdim=True)).square().sum()/
         (hs['before_mix'].double()-hs['before_mix'].double().mean(1,keepdim=True)).square().sum().clamp_min(1e-30)))
    result['limits']=[
        'One stopped checkpoint and one seed; no cross-stage model comparison.',
        'Fixed-coefficient matrix spectra do not equal the full dynamic input Jacobian spectrum.',
        'Across-token covariance effective rank is not a pointwise Jacobian rank.',
        'Frozen probe CE is immediate sensitivity, not evidence of improvement after orthogonally constrained retraining.',
        'Strict orthogonality plus nonnegativity implies a permutation matrix; the signed polar probe leaves the nonnegative Sinkhorn family.',
        'Final mixing cannot create new independent information when its input streams are identical.',
        'Sampled probes use8192 positions with full2K context; paired SE clusters by64 sequences and is descriptive, not an independent retraining significance test.']
    result.update(complete=True,seconds=time.time()-start_time,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    np.savez_compressed(OUT/'mixing-spectra.npz',**spectra.SPECTRA)
    np.savez_compressed(OUT/'mixing-probe-losses.npz',**losses)
    dump('mixing-diagnostics.json',result)
    print(json.dumps(dict(complete=True,seconds=result['seconds'],dev_ce=result['dev_ce'])),flush=True)


if __name__=='__main__':main()
