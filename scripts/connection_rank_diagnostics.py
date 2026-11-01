"""Read-only checkpoint diagnostics on fixed dev rows; no optimizer or W&B.

Token-gradient spectra refer to adjoints of the summed dev token CE with
respect to each token's hidden state. Earlier layers include causal suffix
contributions. They are not a matrix of per-example parameter gradients.
"""
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from nanochat.connection_model import ConnectionConfig, ConnectionModel
from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.packed_training import seed_all
from nanochat.partitioned_readout import vocab_partition

SPECTRA = {}
OUT = Path('results')


def dump(name, data):
    p=OUT/name; p.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def spectral_summary(eigenvalues, name, rows=None):
    lam=eigenvalues.double().clamp_min(0).flip(0)
    SPECTRA[name]=lam.cpu().numpy()
    total=lam.sum(); dim=len(lam)
    if total <= 0:
        return dict(dimension=dim,zero=True,rank_rel_1e6=0,energy_entropy_rank=0.)
    p=lam/total; s=lam.sqrt(); ps=s/s.sum(); cumulative=p.cumsum(0)
    positive=p>0; positive_s=ps>0
    rank6=int((s>s[0]*1e-6).sum()); rank4=int((s>s[0]*1e-4).sum())
    return dict(dimension=dim,sample_rows=rows,zero=False,rank_rel_1e6=rank6,rank_rel_1e4=rank4,
        sigma_min=s[-1].item(),sigma_max=s[0].item(),
        condition_number=(s[0]/s[-1]).item() if s[-1]>0 else None,
        stable_rank=(total/lam[0]).item(),
        energy_entropy_rank=(-torch.sum(p[positive]*p[positive].log())).exp().item(),
        singular_entropy_rank=(-torch.sum(ps[positive_s]*ps[positive_s].log())).exp().item(),
        participation_rank=(1/p.square().sum()).item(),
        top1_energy=p[0].item(),top10_energy=p[:10].sum().item(),
        top512_energy=p[:512].sum().item(),
        dimensions_for_energy={str(q):int(torch.searchsorted(cumulative,q).item()+1) for q in (.9,.95,.99)},
        spectrum_npz_key=name)


def matrix_spectrum(a, name, centered=False):
    a=a.double()
    if centered:a=a-a.mean(0,keepdim=True)
    gram=a.T@a/a.shape[0]
    values=torch.linalg.eigvalsh(gram)
    return spectral_summary(values,name,rows=a.shape[0])


def stream_stats(x):
    # [sample, stream, D], with global sample mean retained for raw cosines.
    x=x.double();n,k,d=x.shape
    flat=x.permute(1,0,2).reshape(k,-1)
    norms=flat.norm(dim=1)
    cos=(flat@flat.T)/(norms[:,None]*norms[None,:]).clamp_min(1e-30)
    unit=x/x.norm(dim=-1,keepdim=True).clamp_min(1e-30)
    token_cos=torch.einsum('nkd,njd->kj',unit,unit)/n
    energy=x.square().sum()
    common=x.mean(1,keepdim=True)
    return dict(flat_cosine=cos.cpu().tolist(),mean_token_cosine=token_cos.cpu().tolist(),
        rms_by_stream=(x.square().mean(dim=(0,2))).sqrt().cpu().tolist(),
        stream_difference_energy_fraction=((x-common).square().sum()/energy.clamp_min(1e-30)).item(),
        aligned_stream_sum_fraction=(x.sum(1).square().sum()/(k*energy).clamp_min(1e-30)).item())


def summarize_site(hidden,gradient,name,targets,partition_inverse):
    h=hidden.cuda().float();g=gradient.cuda().float()
    if h.ndim==2:h=h.unsqueeze(1);g=g.unsqueeze(1)
    r={}
    for label,value in (('activation',h),('gradient',g)):
        a=value.flatten(1)
        r[label]=dict(uncentered=matrix_spectrum(a,name+'_'+label+'_uncentered'),
                      centered=matrix_spectrum(a,name+'_'+label+'_centered',True),
                      streams=stream_stats(value))
    # Gradient row-normalization distinguishes directional diversity from hard-token weighting.
    gn=g.flatten(1);gn=gn/gn.norm(dim=1,keepdim=True).clamp_min(1e-20)
    r['gradient']['unit_row']=matrix_spectrum(gn,name+'_gradient_unit_row')
    if g.shape[1]==4:
        groups=(partition_inverse[targets.cuda()]//8192)
        r['gradient']['energy_by_target_partition_and_stream']=[
            g[groups==i].double().square().sum(-1).mean(0).cpu().tolist() for i in range(4)]
    # Independently inspect the two disjoint halves at the final/head sites.
    if name.endswith('_block8') or name.endswith('_head_normalized'):
        r['gradient']['halves']=[matrix_spectrum(g.flatten(1)[a:b],name+f'_gradient_half{j}',True)
            for j,(a,b) in enumerate(((0,len(g)//2),(len(g)//2,len(g))))]
    return r


def head_geometry(weight, ids, name):
    parts=[];bases=[];polars=[]
    for i,part in enumerate(ids):
        w=weight[part].double()
        # QR + small SVD avoids squaring the condition number when inspecting tiny modes.
        q0,r=torch.linalg.qr(w,mode='reduced')
        u,s,vh=torch.linalg.svd(r,full_matrices=False)
        result=spectral_summary(s.square().flip(0),name+f'_part{i}')
        keep=s>s[0]*1e-10
        q=(q0@u)[:,keep]
        orth_error=(q.T@q-torch.eye(q.shape[1],device=w.device,dtype=w.dtype)).norm().item()
        assert orth_error<1e-6,orth_error
        scale=(w.square().sum()/w.shape[1]).sqrt()
        polar=((q0@u)@vh)*scale
        result['projection_basis_rank_rel_1e10']=int(keep.sum())
        result['polar_completed_missing_directions']=int((~keep).sum())
        polars.append(polar.float());bases.append(q.float())
        result.update(frobenius_norm=w.norm().item(),polar_scale=scale.item(),basis_orthogonality_error=orth_error)
        parts.append(result)
    combined=torch.cat([torch.as_tensor(SPECTRA[name+f'_part{i}'],device=weight.device) for i in range(len(ids))]).sort().values
    return dict(parts=parts,block_diagonal=spectral_summary(combined,name+'_combined')),bases,polars


def head_probe(h,weights,bases,targets,ids,inverse,name,expected_gradient=None):
    """FP32 dense diagnostic on sampled positions, including cap derivative.

    Column-space projection is calculated for the gradient BEFORE the softcap
    input (p-y multiplied by tanh derivative), not the CE gradient alone.
    """
    n,k,d=h.shape
    grouped_targets=inverse[targets]
    losses=[];grads=[];stats=torch.zeros(5,device='cuda',dtype=torch.float64)
    group_target_energy=torch.zeros((k,k),device='cuda',dtype=torch.float64)
    counts=torch.zeros(k,device='cuda',dtype=torch.float64)
    vocab_per=weights[0].shape[0]
    for start in range(0,n,128):
        x=h[start:start+128].float();y=grouped_targets[start:start+128]
        pre=torch.cat([x[:,i]@w.T for i,w in enumerate(weights)],-1)
        tanh=(pre/15).tanh();logits=15*tanh
        losses.append(F.cross_entropy(logits,y,reduction='none').double())
        p=logits.softmax(-1);p[torch.arange(len(x),device='cuda'),y]-=1
        derivative=1-tanh.square();gpre=p*derivative
        gp=[];projected=torch.zeros(len(x),device='cuda')
        for i,(w,q) in enumerate(zip(weights,bases)):
            gi=gpre[:,i*vocab_per:(i+1)*vocab_per]
            gp.append(gi@w)
            projected+=(gi@q).square().sum(-1)
        gp=torch.stack(gp,1);grads.append(gp.cpu())
        energy=gpre.square().sum(-1)
        stats+=torch.stack((p.square().sum().double(),energy.sum().double(),projected.sum().double(),
                           derivative.double().sum(),(derivative<.1).double().sum()))
        group=y//vocab_per
        for i in range(k):
            selected=gp[group==i]
            group_target_energy[i]+=selected.double().square().sum(dim=(0,2));counts[i]+=len(selected)
    gradient=torch.cat(grads).cuda()
    loss=torch.cat(losses)
    r=dict(sample_tokens=n,ce=loss.mean().item(),ce_halves=[v.mean().item() for v in loss.chunk(2)],
        post_softcap_gradient_energy=stats[0].item(),pre_softcap_gradient_energy=stats[1].item(),
        softcap_gradient_energy_ratio=(stats[1]/stats[0]).item(),
        head_column_space_captured_fraction=(stats[2]/stats[1]).item(),
        mean_softcap_derivative=(stats[3]/(n*sum(w.shape[0] for w in weights))).item(),
        fraction_softcap_derivative_below_point1=(stats[4]/(n*sum(w.shape[0] for w in weights))).item(),
        gradient_spectrum=matrix_spectrum(gradient.flatten(1),name+'_gradient'),
        gradient_streams=stream_stats(gradient),
        mean_gradient_energy_by_target_partition=(group_target_energy/counts[:,None].clamp_min(1)).cpu().tolist())
    if expected_gradient is not None:
        ref=expected_gradient.cuda().float()
        if ref.ndim==2:ref=ref.unsqueeze(1)
        r['analytic_vs_cce_gradient_relative_l2']=((gradient-ref).norm()/ref.norm().clamp_min(1e-20)).item()
        assert r['analytic_vs_cce_gradient_relative_l2'] < .02,r['analytic_vs_cce_gradient_relative_l2']
    assert -1e-5<=r['head_column_space_captured_fraction']<=1.00001
    return r


def self_test():
    torch.manual_seed(51)
    a=torch.randn(40,8,device='cuda',dtype=torch.float64)
    result=matrix_spectrum(a,'self_test')
    expected=torch.linalg.svdvals(a).square()/len(a)
    actual=torch.as_tensor(SPECTRA['self_test'],device='cuda')
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)
    q,_=torch.linalg.qr(a)
    flat=matrix_spectrum(q,'self_test_flat')
    assert abs(flat['energy_entropy_rank']-8)<1e-8
    identical=a[:,None,:].expand(-1,4,-1)
    assert stream_stats(identical)['stream_difference_energy_fraction']==0
    # Analytic head gradient is checked against independent autograd in FP64.
    h=torch.randn(11,4,8,device='cuda',dtype=torch.float64,requires_grad=True)
    w=torch.randn(4,16,8,device='cuda',dtype=torch.float64)
    z=torch.cat([h[:,i]@w[i].T for i in range(4)],-1)
    y=torch.arange(11,device='cuda');cap=15*(z/15).tanh()
    expected_grad=torch.autograd.grad(F.cross_entropy(cap,y,reduction='sum'),h)[0]
    gp=cap.softmax(-1);gp[torch.arange(11,device='cuda'),y]-=1
    gp=gp*(1-(z/15).tanh().square())
    analytic=torch.stack([gp[:,i*16:(i+1)*16]@w[i] for i in range(4)],1)
    torch.testing.assert_close(analytic,expected_grad,atol=1e-12,rtol=1e-10)
    dump('self-test.json',dict(passed=True,svd_oracle=True,equal_spectrum=True,identical_streams=True,analytic_gradient=True))


def run_case(case,dev,sample_positions,common_inverse):
    seed_all(42)
    checkpoint=torch.load(case['path'],map_location='cpu',weights_only=False)
    assert checkpoint['run_id']==case['run_id']
    assert checkpoint['step']==case['expected_step']
    c=ConnectionConfig.from_checkpoint(checkpoint['config'])
    assert c.variant=='mhc' and c.architecture=='vanilla' and c.width==512 and c.streams==4
    assert c.readout==case['readout']
    model=ConnectionModel(c).prepare(compile_blocks=True)
    match_vanilla_initialization(model,42)
    ids=model.partition_ids if c.readout=='vocab_partition' else torch.arange(c.vocab_size,device='cuda')[None,:]
    initial,_,_=head_geometry(model.lm_head.weight.detach().bfloat16().float(),ids,case['label']+'_initial_head')
    model.load_state_dict(checkpoint['model'],strict=True);del checkpoint;gc.collect()
    model.eval()
    ids=model.partition_ids if c.readout=='vocab_partition' else torch.arange(c.vocab_size,device='cuda')[None,:]
    inverse=model.partition_inverse if c.readout=='vocab_partition' else torch.arange(c.vocab_size,device='cuda')
    geometry,bases,polars=head_geometry(model.lm_head.weight.detach().bfloat16().float(),ids,case['label']+'_head')
    capture={};hidden_samples={};gradient_samples={};sample_targets=[]
    for i,block in enumerate(model.blocks):
        original=block.forward
        def traced(*args,index=i,forward=original,**kwargs):
            if index==0:
                capture['expanded_embedding']=args[0];args[0].retain_grad()
            out=forward(*args,**kwargs);out.retain_grad();capture[f'block{index+1}']=out
            return out
        block.forward=traced
    original_hidden=model.hidden
    def traced_hidden(*args,**kwargs):
        h=original_hidden(*args,**kwargs);h.retain_grad();capture['head_normalized']=h
        return h
    model.hidden=traced_hidden
    aggregate_head_gradient=torch.zeros_like(model.lm_head.weight,dtype=torch.float32)
    total_ce=0.;total_tokens=0
    # Read exactly the same64 dev contexts, selecting128 positions/context.
    for start in range(0,64,2):
        row=torch.from_numpy(np.array(dev[start:start+2],dtype=np.int64,copy=True)).cuda()
        x,y=row[:,:-1].contiguous(),row[:,1:].contiguous()
        model.zero_grad(set_to_none=True);capture.clear()
        loss=model(x,y);loss.backward()
        total_ce+=loss.item()*y.numel();total_tokens+=y.numel()
        aggregate_head_gradient+=model.lm_head.weight.grad.float()*(y.numel()/(64*2048))
        pos=torch.as_tensor(sample_positions[start:start+2],device='cuda')
        batch=torch.arange(2,device='cuda')[:,None]
        sample_targets.append(y[batch,pos].reshape(-1).cpu())
        for name,value in capture.items():
            a=value.detach()[batch,pos]
            b=value.grad.detach()[batch,pos].float()*y.numel()
            if name!='head_normalized':a=a.transpose(-1,-2);b=b.transpose(-1,-2)
            hidden_samples.setdefault(name,[]).append(a.reshape(-1,*a.shape[2:]).cpu())
            gradient_samples.setdefault(name,[]).append(b.reshape(-1,*b.shape[2:]).cpu())
        if start%16==0:print(json.dumps(dict(case=case['label'],dev_rows=start+2,loss=loss.item())),flush=True)
    raw_h={name:torch.cat(v) for name,v in hidden_samples.items()}
    raw_g={name:torch.cat(v) for name,v in gradient_samples.items()}
    targets=torch.cat(sample_targets)
    del hidden_samples,gradient_samples;capture.clear();model.zero_grad(set_to_none=True)
    r=dict(label=case['label'],checkpoint_step=case['expected_step'],training_tokens=case['expected_step']*32768,
           readout=c.readout,dev_ce=total_ce/total_tokens,dev_tokens=total_tokens,sampled_tokens=len(targets),
           initial_head=initial,trained_head=geometry,sites={},head_parameter_gradient={})
    for name in raw_h:
        r['sites'][name]=summarize_site(raw_h[name],raw_g[name],case['label']+'_'+name,targets,common_inverse)
        print(json.dumps(dict(case=case['label'],site=name,
            gradient_effective_rank=r['sites'][name]['gradient']['uncentered']['energy_entropy_rank'],
            activation_effective_rank=r['sites'][name]['activation']['centered']['energy_entropy_rank'])),flush=True)
        dump(case['label']+'-partial.json',r)
    for i,part in enumerate(ids):
        r['head_parameter_gradient'][str(i)]=matrix_spectrum(aggregate_head_gradient[part],case['label']+f'_head_parameter_gradient{i}')
    hn=raw_h['head_normalized'].cuda()
    if hn.ndim==2:hn=hn.unsqueeze(1)
    weights=[model.lm_head.weight.detach()[part].bfloat16().float() for part in ids]
    r['head_probes']={}
    r['head_probes']['original']=head_probe(hn,weights,bases,targets.cuda(),ids,inverse,
        case['label']+'_probe_original',raw_g['head_normalized'])
    r['head_probes']['polar_same_frobenius']=head_probe(hn,[p.bfloat16().float() for p in polars],bases,
        targets.cuda(),ids,inverse,case['label']+'_probe_polar')
    if c.readout=='vocab_partition':
        # Forward-only counterfactuals, no training and no checkpoint mutation.
        common=raw_h['block8'].cuda().mean(1,keepdim=True).expand(-1,4,-1)
        common=F.rms_norm(common,(c.width,))
        r['head_probes']['mean_streams']=head_probe(common,weights,bases,targets.cuda(),ids,inverse,case['label']+'_probe_mean')
        r['head_probes']['cycle_stream_assignment']=head_probe(hn.roll(1,dims=1),weights,bases,targets.cuda(),ids,inverse,case['label']+'_probe_cycle')
    r['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    dump(case['label']+'.json',r)
    del model,raw_h,raw_g,aggregate_head_gradient,bases,polars;gc.collect();torch.cuda.empty_cache()
    return r


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    configure_attention('sdpa_cudnn');seed_all(42)
    assert torch.cuda.device_count()==1
    torch.cuda.reset_peak_memory_stats()
    self_test()
    plan=json.loads(Path('../diagnostic-inputs.json').read_text())
    manifest=json.loads(Path('../data/manifest.json').read_text())
    dev_path=Path('../data/dev.bin')
    assert hashlib.sha256(dev_path.read_bytes()).hexdigest()==manifest['files']['dev']['sha256']
    dev=np.memmap(dev_path,mode='r',dtype=np.uint16,shape=(64,2049))
    rng=np.random.default_rng(20261031)
    positions=np.stack([np.sort(rng.choice(2048,128,replace=False)) for _ in range(64)])
    _,common_inverse=vocab_partition(32768,4,20261031);common_inverse=common_inverse.cuda()
    start=time.time();cases={}
    for case in plan['cases']:
        cases[case['label']]=run_case(case,dev,positions,common_inverse)
    np.savez_compressed(OUT/'spectra.npz',**SPECTRA)
    result=dict(complete=True,logging_backend='local',gpu=torch.cuda.get_device_name(),
        plan=plan,dev_sha256=manifest['files']['dev']['sha256'],sample_positions_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),
        sample_definition='64 heldout dev sequences x128 fixed uniformly selected positions; full2K context and full-token CE backward',
        effective_rank_definition='Primary: exp(entropy of normalized squared singular values); also singular-value entropy, stable and participation ranks. Centered and uncentered separately.',
        gradient_definition='Rows are token hidden-state adjoints of summed CE, before any optimizer; earlier layers include suffix contributions. Head parameter gradient separately aggregates all131072 dev tokens.',
        limits=['Different training steps in interim comparison; final phase compares equal10B checkpoints.',
            'Across-token gradient covariance rank is not a pointwise Jacobian rank. Input-dependent normalization/mixing can rotate gradients across tokens.',
            'Covariance rank and stream correlation are descriptive; do not identify a unique causal training failure.',
            'Frozen polar/mean/permutation probes are immediate forward sensitivities, not training experiments.',
            'No gradient loss claim for whole model: components outside current head column space still update head parameters.'],
        seconds=time.time()-start,cases=cases)
    dump('diagnostics.json',result)
    print(json.dumps(dict(complete=True,seconds=result['seconds'],cases=list(cases))),flush=True)


if __name__=='__main__':main()
