"""Deterministic, tensor-free planning for FFN structural reallocation."""
import math


def form_schedule(total_steps, count=10):
    steps = [k * total_steps // (count + 1) for k in range(1, count + 1)]
    if count < 1 or len(set(steps)) != count or not 0 < steps[0] <= steps[-1] < total_steps:
        raise ValueError('Form needs distinct update boundaries strictly inside training')
    return steps


def form_budget(total_width, depth, hidden_dim, count=10):
    surplus = total_width - depth * hidden_dim
    # Nearest multiple of 64, ties upward; computed once from initial widths.
    budget = ((surplus + 32 * count) // (64 * count)) * 64
    if not 0 < budget <= surplus or 2 * budget > total_width:
        raise ValueError('Invalid Form budget for these model dimensions')
    return budget


def round_growth(counts, budget):
    if sum(counts) != budget or budget % 64 or any(n < 0 for n in counts):
        raise ValueError('Growth counts must sum to a positive 64-aligned budget')
    growth = [n // 64 * 64 for n in counts]
    remainder_order = sorted(range(len(counts)), key=lambda i: (-(counts[i] % 64), i))
    for i in remainder_order[:(budget - sum(growth)) // 64]:
        growth[i] += 64
    assert sum(growth) == budget
    return growth


def plan_reallocation(scores, hidden_dim, budget, rng=None):
    """Greedy globally minimal pruning with per-layer capacity, then top retained.

    Ties use (layer, channel). Random control shuffles the same eligible channel
    populations independently for pruning and growth, with an isolated RNG.
    Only growth quotas are 64-aligned; pruning is genuinely per-channel.
    """
    widths = [len(s) for s in scores]
    if any(m < hidden_dim for m in widths) or budget <= 0 or budget % 64:
        raise ValueError('Invalid widths or Form budget')
    if sum(m - hidden_dim for m in widths) < budget or sum(widths) < 2 * budget:
        raise ValueError('Insufficient eligible channels')
    entries = [(float(s), l, j) for l, layer in enumerate(scores) for j, s in enumerate(layer)]
    if not all(math.isfinite(s) for s, _, _ in entries):
        raise FloatingPointError('Nonfinite Form importance')
    ranked = sorted(entries)
    if rng is not None:
        rng.shuffle(ranked)
    removed = [set() for _ in widths]
    remaining = budget
    for _, l, j in ranked:
        if len(removed[l]) < widths[l] - hidden_dim:
            removed[l].add(j)
            remaining -= 1
            if not remaining:
                break
    retained = [e for e in entries if e[2] not in removed[e[1]]]
    retained.sort(key=lambda e: (-e[0], e[1], e[2]))
    if rng is not None:
        rng.shuffle(retained)
    donors = retained[:budget]
    counts = [0] * len(widths)
    for _, l, _ in donors:
        counts[l] += 1
    growth = round_growth(counts, budget)
    keep = [[j for j in range(m) if j not in removed[l]] for l, m in enumerate(widths)]
    after = [len(k) + g for k, g in zip(keep, growth)]
    assert sum(after) == sum(widths) and min(after) >= hidden_dim
    return dict(widths_before=widths, widths_after=after, keep=keep,
                prune=[sorted(r) for r in removed], growth=growth, donor_counts=counts,
                donors=[[l, j] for _, l, j in donors])
