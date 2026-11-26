"""Published Union optimization within the fixed NanoChat pretraining budget."""
import math

import numpy as np
import torch

from nanochat.packed_training import train_step
from nanochat.procedural import REFERENCE_SYMBOLS, transform_example


class ReferenceAdamW(torch.optim.AdamW):
    """AdamW with FP32 optimizer weights and the released global gradient clip."""

    def __init__(self, model):
        self.low_precision = []
        parameters = []
        for weight in model.parameters():
            if weight.dtype == torch.float32:
                parameters.append(weight)
            else:
                master = torch.nn.Parameter(weight.detach().float())
                self.low_precision.append((weight, master))
                parameters.append(master)
        super().__init__(parameters, lr=5e-4, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8)

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for weight, _ in self.low_precision:
            weight.grad = None

    @torch.no_grad()
    def step(self):
        for weight, master in self.low_precision:
            master.grad = None if weight.grad is None else weight.grad.float()
        torch.nn.utils.clip_grad_norm_(self.param_groups[0]['params'], 1., error_if_nonfinite=True)
        super().step()
        for weight, master in self.low_precision:
            weight.copy_(master)

    def state_dict(self):
        return {**super().state_dict(), 'master_weights': [m.detach() for _, m in self.low_precision]}

    @torch.no_grad()
    def load_state_dict(self, state):
        saved = state['master_weights']
        if len(saved) != len(self.low_precision):
            raise ValueError('Reference AdamW checkpoint has incompatible master weights')
        super().load_state_dict({k: v for k, v in state.items() if k != 'master_weights'})
        for (weight, master), value in zip(self.low_precision, saved):
            master.copy_(value)
            weight.copy_(master)


class ReferenceUnion:
    """64 independently seeded examples per AdamW update, packed into full baseline rows."""

    batch_size = 64
    source_epoch_updates = 2501

    def __init__(self, task, updates, budget_steps, rows_per_step):
        if task.task != 'union' or not budget_steps <= updates <= budget_steps * rows_per_step:
            raise ValueError('Reference Union needs one or more updates per budget step and at least one row per update')
        if updates > self.source_epoch_updates:
            raise ValueError('Reference Union supports one released training epoch (up to 2501 updates)')
        self.task = task
        self.updates = updates
        self.budget_steps = budget_steps
        self.rows_per_step = rows_per_step
        rng = torch.Generator().manual_seed(task.seed)
        self.order = torch.randperm(self.batch_size * self.source_epoch_updates, generator=rng)

    def completed(self, budget_step):
        return budget_step * self.updates // self.budget_steps

    def batch(self, update, rows):
        task = self.task
        pad, sep = task.vocab_size - 1, task.vocab_size - 2
        x = np.full((rows, task.context), pad, dtype=np.int64)
        y = np.full_like(x, pad)
        mask = np.zeros(x.shape, dtype=bool)
        used = [0] * rows
        first = update * self.batch_size
        samples = []
        for index in self.order[first:first+self.batch_size].tolist():
            rng = np.random.RandomState((task.seed + index) % 2**32)
            samples.append(transform_example('union', task.length, REFERENCE_SYMBOLS, sep, rng))
        # Longest-first packing balances the rows without dropping any example.
        # Row assignment has no effect on a document-isolated, answer-only objective.
        for tokens, answer in sorted(samples, key=lambda s: -len(s[0])):
            row = min(range(rows), key=used.__getitem__)
            pos = used[row]
            end = pos + len(tokens)
            if end > task.context:
                raise ValueError('64 whole Union examples exceed the configured compute budget')
            x[row, pos:end] = tokens
            y[row, pos:end-1] = tokens[1:]
            mask[row, pos+answer-1:end-1] = True
            used[row] = end + 1
        return x, y, mask

    def train(self, model, optimizer, budget_step, class_ids, microbatch, gradient_stats):
        start, end = self.completed(budget_step), self.completed(budget_step + 1)
        weighted_loss = 0.
        targets = 0
        for i in range(end-start):
            # Across this group of optimizer updates the model processes exactly
            # the same number of full-length rows as a vanilla optimizer step.
            rows = (i+1)*self.rows_per_step//(end-start) - i*self.rows_per_step//(end-start)
            data = self.batch(start+i, rows)
            count = int(data[2].sum())
            stats = {} if gradient_stats is not None else None
            loss = train_step(model, optimizer, None, start+i, seed=self.task.seed,
                              microbatch=rows if rows <= microbatch else math.gcd(rows, microbatch), batch_tokens=rows*self.task.context,
                              masked_batch=data, class_ids=class_ids, gradient_stats=stats)
            if stats is not None:
                gradient_stats['gradient_norm'] = max(gradient_stats.get('gradient_norm', 0.), stats['gradient_norm'])
            weighted_loss += count * loss
            targets += count
        return weighted_loss / targets, targets
