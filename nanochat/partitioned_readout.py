"""Disjoint stream/vocabulary readout with ONE full-vocabulary softmax.

Each CCE invocation returns a differentiable local log-partition. Combining
these with logsumexp preserves denominator gradients to every stream. No
token is dropped, no local softmax objective is averaged, and no gradient
filtering is enabled. Uses the public API of the pinned Apple CCE revision.
"""
import torch
from torch.nn import functional as F


def vocab_partition(vocab_size, streams, seed):
    if vocab_size % streams:
        raise ValueError('The balanced partition requires vocab_size divisible by streams')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    grouped_ids = torch.randperm(vocab_size, generator=generator)
    return grouped_ids.reshape(streams, -1), torch.argsort(grouped_ids)


def partitioned_logits(hidden, weight, partition_ids, inverse):
    """hidden [..., K, D]; weights remain indexed by original token ID."""
    logits = torch.cat([F.linear(hidden[..., i, :], weight[ids])
                        for i, ids in enumerate(partition_ids)], dim=-1)
    return 15 * torch.tanh(logits[..., inverse].float() / 15)


def combine_partition_losses(losses, log_partitions, target_stream):
    """Exact global CE from local CE/LSE; also independently tested in FP64."""
    local_lse = torch.stack(log_partitions, dim=-1)
    negative_target = torch.stack(losses, dim=-1) - local_lse
    negative_target = negative_target.gather(-1, target_stream.unsqueeze(-1)).squeeze(-1)
    return (torch.logsumexp(local_lse, dim=-1) + negative_target).mean()


def partitioned_cross_entropy(hidden, weight, targets, partition_ids, inverse):
    from cut_cross_entropy import linear_cross_entropy
    group_size = partition_ids.shape[1]
    grouped_targets = inverse[targets]
    target_stream = grouped_targets // group_size
    # A valid dummy local target is required in non-target partitions. Its
    # local CE is discarded, but its differentiable LSE is kept globally.
    local_target = grouped_targets % group_size
    losses, log_partitions = [], []
    for i, ids in enumerate(partition_ids):
        loss, lse = linear_cross_entropy(hidden[..., i, :], weight[ids], local_target,
            softcap=15., shift=0, reduction='none', return_lse=True,
            filter_eps=None, filter_e_grad=False, filter_c_grad=False,
            accum_e_fp32=True, accum_c_fp32=True)
        losses.append(loss)
        log_partitions.append(lse)
    return combine_partition_losses(losses, log_partitions, target_stream)
