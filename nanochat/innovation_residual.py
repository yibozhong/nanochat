"""Affine residual prediction with strictly separated local and NTP gradients."""
import torch
from torch import nn
from torch.nn import functional as F


class InnovationResidual(nn.Module):
    def __init__(self, width):
        super().__init__()
        # Full-rank at initialization, near zero; no random draws or rank constraint.
        self.weight = nn.Parameter(1e-3 * torch.eye(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, h, residual, beta, local_loss=False, collect_statistics=False):
        # Freeze parameters, NOT the output: NTP still differentiates -beta*W*h.
        pred = F.linear(h, self.weight.detach().to(h.dtype), self.bias.detach().to(h.dtype))
        innovation = residual - beta.to(h.dtype) * pred
        updated = h + innovation
        loss = h.new_zeros((), dtype=torch.float32)
        if local_loss:
            local_pred = F.linear(h.detach(), self.weight.to(h.dtype), self.bias.to(h.dtype))
            loss = (local_pred.float() - residual.detach().float()).square().mean()
        stats = h.new_empty((0,), dtype=torch.float64)
        if collect_statistics:
            stats = prediction_sums(h, residual, pred, innovation)
        return updated, loss, stats


@torch.no_grad()
def prediction_sums(h, residual, pred, innovation):
    """Additive sufficient statistics; FP64 reductions only for evaluation metrics."""
    r = residual.detach().reshape(-1, residual.shape[-1]).double()
    p = pred.detach().reshape_as(r).double()
    return torch.cat((r.sum(0), r.square().sum(0), torch.stack((
        (p-r).square().sum(), p.square().sum(), innovation.double().square().sum(),
        h.double().square().sum(), r.new_tensor(r.shape[0])))))


def summarize_prediction_sums(sums, width):
    """Variance-weighted multi-output R², centered per feature over all dev tokens."""
    rows = []
    for index, row in enumerate(sums.detach().cpu().double()):
        target_sum, target_sq = row[:width], row[width:2*width]
        sse, p2, i2, h2, count = row[2*width:].tolist()
        r2 = target_sq.sum().item()
        sst = max(0., r2 - target_sum.square().sum().item()/count)
        rows.append(dict(layer=index//2, branch='attention' if index%2==0 else 'mlp',
            normalized_depth=(index//2+1)/(len(sums)//2), tokens=int(count),
            mse=sse/(count*width), r_squared=1-sse/sst if sst>1e-20 else None,
            target_centered_sum_squares=sst, target_sum_squares=r2,
            prediction_to_residual_norm=(p2/r2)**.5 if r2>0 else None,
            innovation_to_residual_norm=(i2/r2)**.5 if r2>0 else None,
            residual_to_hidden_norm=(r2/h2)**.5 if h2>0 else None,
            innovation_to_hidden_norm=(i2/h2)**.5 if h2>0 else None))
    return rows
