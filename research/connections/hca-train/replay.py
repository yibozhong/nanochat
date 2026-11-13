"""Diagnostic-only CCE gradient recording for exact checkpoint replay.

The installed CCE has nondeterministic parallel forward/backward reductions. A real
production update records its CCE inputs, output and gradients. Replay requires
identical inputs/targets/scaling and supplies exactly those recorded gradients;
the rest of the model backward and optimizer update execute normally. This is
never imported by formal training and is not a deterministic CCE implementation.
"""
from contextlib import contextmanager


@contextmanager
def cce_tape(records=None, *, actual=False):
    import torch
    import cut_cross_entropy
    original = cut_cross_entropy.linear_cross_entropy
    capture = records is None
    assert not actual or not capture, 'Actual replay requires captured inputs'
    entries = [] if capture else records
    index = 0

    def save(value):
        return value.detach().cpu().clone()

    def check_inputs(h, w, targets, entry):
        for name, value in (('h', h), ('w', w), ('targets', targets)):
            expected = entry[name]
            assert value.dtype == expected.dtype and torch.equal(value.detach().cpu(), expected), name

    class Replay(torch.autograd.Function):
        @staticmethod
        def forward(ctx, h, w, targets, entry):
            check_inputs(h, w, targets, entry)
            ctx.entry = entry
            ctx.device = h.device
            return entry['loss'].to(h.device).clone()

        @staticmethod
        def backward(ctx, gradient):
            entry = ctx.entry
            # Captured gradients already include the accumulation loss scale.
            # Require the same scale, rather than scaling a rounded BF16 value
            # a second time or relying on any numerical tolerance.
            assert torch.equal(gradient.detach().cpu(), entry['loss_gradient']), 'Loss scale changed'
            return entry['h_gradient'].to(ctx.device), entry['w_gradient'].to(ctx.device), None, None

    def wrapped(h, w, targets, **kwargs):
        nonlocal index
        if capture:
            entry = dict(h=save(h), w=save(w), targets=save(targets), kwargs=kwargs.copy())
            loss = original(h, w, targets, **kwargs)
            entry['loss'] = save(loss)
            for name, value in (('h', h), ('w', w), ('loss', loss)):
                def hook(gradient, name=name, entry=entry):
                    entry[name+'_gradient'] = save(gradient)
                value.register_hook(hook)
            entries.append(entry)
        else:
            assert index < len(entries), 'Extra CCE call after restore'
            entry = entries[index]
            assert kwargs == entry['kwargs'], 'CCE arguments changed'
            if actual:
                # Establish identical model outputs BEFORE attributing a loss
                # discrepancy to CCE's unordered locked logaddexp reduction.
                check_inputs(h, w, targets, entry)
                loss = original(h, w, targets, **kwargs)
            else:
                loss = Replay.apply(h, w, targets, entry)
        index += 1
        return loss

    cut_cross_entropy.linear_cross_entropy = wrapped
    try:
        yield entries
        assert index == len(entries) and index > 0
        assert all(all(name+'_gradient' in entry for name in ('h', 'w', 'loss')) for entry in entries)
    finally:
        cut_cross_entropy.linear_cross_entropy = original
