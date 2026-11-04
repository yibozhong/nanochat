"""Post-optimizer spectral equalization of attention output weights only.

Uses the five Polar Express polynomials already used by this fork's Muon.
Finite BF16 iterations approximate a polar factor; they are not an exact
Stiefel projection, and cannot invent missing singular directions.
"""
import time
import torch
from nanochat.optim import polar_express_coeffs


def scaled_ns_projection(weights, work_dtype=torch.bfloat16):
    """Apply NS to [...,D,D], then preserve each matrix's original Frobenius norm."""
    target = weights.norm(dim=(-2,-1), keepdim=True)
    x = weights.to(work_dtype)
    # Same normalization and five coefficients as the native Muon NS stage.
    x = x / (x.norm(dim=(-2,-1), keepdim=True)*1.01 + 1e-6)
    for a,b,c in polar_express_coeffs:
        gram = x @ x.mT
        polynomial = b*gram + c*(gram @ gram)
        x = a*x + polynomial @ x
    direction = x.to(weights.dtype)
    current = direction.norm(dim=(-2,-1), keepdim=True).clamp_min(1e-30)
    return direction*(target/current)


@torch.no_grad()
@torch.compile(dynamic=False, fullgraph=True)
def project_attention_outputs_(parameters):
    weights = torch.stack(parameters)
    projected = scaled_ns_projection(weights)
    torch._foreach_copy_(parameters, projected.unbind(0))


def install_attention_output_ns(model, optimizer):
    """One projection after each complete step, including after all accumulation.

No momentum transport/reset and no backward through the projection. A fresh
optimizer installs the hook before loading state, so resume needs no extra state.
    """
    parameters = tuple(block.attn.c_proj.weight for block in model.blocks)
    assert all(p.ndim==2 and p.shape[0]==p.shape[1] for p in parameters)
    assert not hasattr(optimizer, '_attention_output_ns_handle')
    def after_step(opt, args, kwargs):
        project_attention_outputs_(parameters)
    optimizer._attention_output_ns_handle = optimizer.register_step_post_hook(after_step)


def spectrum_summary(singular_values):
    s=singular_values.double()
    energy=s.square();total=energy.sum()
    if total.item()==0:
        return dict(energy_erank=0.,stable_rank=0.,numerical_rank=0,condition=None,
                    minimum=0.,maximum=0.,frobenius=0.,scaled_orthogonality_error=None)
    p=energy/total
    return dict(energy_erank=float(torch.exp(-(p*p.clamp_min(1e-300).log()).sum())),
        stable_rank=float(total/energy.max()),numerical_rank=int((s>s.max()*1e-6).sum()),
        condition=float(s.max()/s.min()) if s.min()>0 else None,
        minimum=float(s.min()),maximum=float(s.max()),frobenius=float(total.sqrt()),
        scaled_orthogonality_error=float((energy/energy.mean()-1).square().mean().sqrt()))


@torch.no_grad()
def attention_output_diagnostics(model, data, sequences=4, positions=128):
    """Fixed dev token sample; spectra of W_o, pre-W_o and post-W_o features.

Read-only eager block traversal uses the same attention kernel. Hooks and SVD
stay outside compiled training. No labels, optimizer steps or RNG draws here.
    """
    start=time.perf_counter();was_training=model.training;compiled=model.compiled_steps
    captured={};handles=[]
    n=min(sequences,len(data.arrays['dev']))
    def capture(index):
        def hook(module,args,output):
            selection=torch.linspace(0,output.shape[1]-1,min(positions,output.shape[1]),
                                     device=output.device).long()
            captured[index]=(args[0][:,selection].flatten(0,1).double(),
                             output[:,selection].flatten(0,1).double())
        return hook
    try:
        model.eval();model.compiled_steps=None
        for i,block in enumerate(model.blocks):handles.append(block.attn.c_proj.register_forward_hook(capture(i)))
        x,_=data.batch('dev',0,n);model.hidden(x)
    finally:
        for handle in handles:handle.remove()
        model.compiled_steps=compiled;model.train(was_training)
    assert len(captured)==len(model.blocks)
    layers=[]
    for i,block in enumerate(model.blocks):
        weight=block.attn.c_proj.weight.detach().double()
        s=torch.linalg.svdvals(weight)
        row=dict(layer=i,normalized_depth=(i+1)/len(model.blocks),weight=spectrum_summary(s),
                 weight_singular_values=s.cpu().tolist())
        for name,features in zip(('input_features','output_features'),captured[i]):
            centered=features-features.mean(0,keepdim=True)
            row[name]=dict(centered=spectrum_summary(torch.linalg.svdvals(centered)),
                           uncentered=spectrum_summary(torch.linalg.svdvals(features)),
                           sampled_tokens=features.shape[0])
        layers.append(row)
    return dict(layers=layers,sequences=n,positions_per_sequence=min(positions,model.config.sequence_length),
        token_selection='fixed evenly spaced positions over complete dev contexts; no RNG',
        interpretation='Weight orthogonality does not whiten input covariance; feature eRank is a cross-token statistic, not a pointwise Jacobian rank.',
        seconds=time.perf_counter()-start)
