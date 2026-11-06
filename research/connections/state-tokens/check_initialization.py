"""Bounded CPU initialization audit, not language-model training or a LR sweep."""
import json
import math
from pathlib import Path

import torch

from nanochat.state_tokens import StateTokenResidual
from nanochat.gpt import norm


def main():
    torch.set_num_threads(1)
    records = []
    for margin in (8., 10., 12.):
        for seed in (42, 1337, 2026):
            torch.manual_seed(seed)
            module = StateTokenResidual(512, 4, 'math')
            with torch.no_grad():
                module.write.q.weight.mul_(math.sqrt(margin/10))
                module.write.k.weight.mul_(math.sqrt(margin/10))
            embedding = norm(torch.randn(2, 32, 512)).requires_grad_()
            state = module.initialize(embedding)
            probe = torch.randn_like(embedding)
            for _ in range(16):
                hidden, state = module(torch.zeros_like(embedding), state)
            gradient, = torch.autograd.grad(hidden, embedding, probe)
            row = dict(margin=margin, seed=seed,
                embedding_error=((hidden-embedding).norm()/embedding.norm()).item(),
                slot_rms=(state-state.mean(-2, keepdim=True)).square().mean().sqrt().item(),
                gradient_error=((gradient-probe).norm()/probe.norm()).item())
            row['passed'] = row['embedding_error']<.1 and row['slot_rms']>.25 and row['gradient_error']<.2
            records.append(row); print(json.dumps(row), flush=True)
    passed = [m for m in (8.,10.,12.) if all(r['passed'] for r in records if r['margin']==m)]
    report = dict(records=records, selected_margin=min(passed) if passed else None,
                  criteria=dict(embedding_relative_error=.1, min_slot_rms=.25, gradient_relative_error=.2))
    Path('results/initialization-selection.json').write_text(json.dumps(report, indent=2)+'\n')
    assert passed, report


if __name__ == '__main__': main()
