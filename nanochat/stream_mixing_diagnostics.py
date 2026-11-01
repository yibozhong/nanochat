"""Small, fixed-input diagnostics of the final mixer; no updates or logging."""
import torch


@torch.no_grad()
def inspect_stream_mixing(model, idx):
    capture = {}
    def collect(module, args, output):
        capture['before'] = args[0].detach()
        capture['after'] = output.detach()
    hook = model.partition_mixer.register_forward_hook(collect)
    try:
        model.hidden(idx)
    finally:
        hook.remove()
    matrix = model.partition_mixer.mixing_matrix(capture['before']).double()
    singular = torch.linalg.svdvals(matrix)
    def quantiles(x):
        return {str(q):torch.quantile(x.flatten(),q).item() for q in (0.,.5,1.)}
    result = dict(tokens=idx.numel(),matrix_mean=matrix.mean((0,1)).cpu().tolist(),
        minimum_singular_value=quantiles(singular[...,-1]),
        condition_number=quantiles(singular[...,0]/singular[...,-1].clamp_min(1e-30)),
        full_rank_fraction_rel_1e6=(singular[...,-1]>1e-6*singular[...,0]).double().mean().item(),
        row_sum_max_error=(matrix.sum(-1)-1).abs().max().item(),
        column_sum_max_error=(matrix.sum(-2)-1).abs().max().item(),
        rank_definition='Rank of the4x4 coefficient matrix with coefficients held fixed; not the full input-dependent Jacobian')
    for name,x in capture.items():
        x=x.double().reshape(-1,x.shape[-2],x.shape[-1]).transpose(-1,-2)
        unit=x/x.norm(dim=-1,keepdim=True).clamp_min(1e-30)
        result[name]=dict(mean_token_cosine=torch.einsum('nkd,njd->kj',unit,unit).div(len(x)).cpu().tolist(),
            stream_difference_energy_fraction=((x-x.mean(1,keepdim=True)).square().sum()/x.square().sum().clamp_min(1e-30)).item())
    return result
