"""Small, fixed-input diagnostics of the final mixer; no updates or logging."""
import torch


@torch.no_grad()
def inspect_stream_mixing(model, idx):
    before, after = model.hidden(idx, return_streams=True)
    capture = dict(before=before.detach(), after=after.detach())
    if model.partition_mixer is None:
        matrix=torch.eye(4,device=idx.device,dtype=torch.float64).expand(*idx.shape,4,4)
    else:
        matrix = model.partition_mixer.mixing_matrix(capture['before']).double()
    singular = torch.linalg.svdvals(matrix)
    def quantiles(x):
        return {str(q):torch.quantile(x.flatten(),q).item() for q in (0.,.5,1.)}
    result = dict(tokens=idx.numel(),partition_mixing=model.config.partition_mixing,matrix_mean=matrix.mean((0,1)).cpu().tolist(),
        minimum_singular_value=quantiles(singular[...,-1]),
        condition_number=quantiles(singular[...,0]/singular[...,-1].clamp_min(1e-30)),
        full_rank_fraction_rel_1e6=(singular[...,-1]>1e-6*singular[...,0]).double().mean().item(),
        row_sum_max_error=(matrix.sum(-1)-1).abs().max().item(),
        column_sum_max_error=(matrix.sum(-2)-1).abs().max().item(),
        orthogonality_max_error=(matrix.transpose(-1,-2)@matrix-torch.eye(4,device=idx.device,dtype=torch.float64)).abs().max().item(),
        rank_definition='Rank of the4x4 coefficient matrix with coefficients held fixed; not the full input-dependent Jacobian')
    difference_energy={}
    for name,x in capture.items():
        x=x.double().reshape(-1,x.shape[-2],x.shape[-1]).transpose(-1,-2)
        unit=x/x.norm(dim=-1,keepdim=True).clamp_min(1e-30)
        result[name]=dict(mean_token_cosine=torch.einsum('nkd,njd->kj',unit,unit).div(len(x)).cpu().tolist(),
            stream_difference_energy_fraction=((x-x.mean(1,keepdim=True)).square().sum()/x.square().sum().clamp_min(1e-30)).item())
        difference_energy[name]=(x-x.mean(1,keepdim=True)).square().sum().item()
    result['stream_difference_energy_retention']=(difference_energy['after']/difference_energy['before']
        if difference_energy['before']>0 else None)
    return result
