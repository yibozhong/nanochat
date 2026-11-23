"""Completed block displacement memory, including endpoint gradients and grouping."""
from dataclasses import replace

import pytest
import torch

from test_chord_attnres import setting
from test_chord_history_correction import check_training_and_resume
from test_momentum import gpu, inputs, make


@pytest.mark.parametrize('backend', ['direct', 'packed'])
def test_update_history_training_and_resume(setting, backend, tmp_path):
    # The common check covers compiled zero-start parity, gate/scorer opening,
    # KV caching, causality, and the next update after an optimizer restart.
    check_training_and_resume(setting, backend, tmp_path, updates=True)


@pytest.mark.parametrize('backend,size', [('direct', 2), ('packed', 2), ('direct', 4)])
def test_block_values_and_endpoint_gradients(setting, backend, size, monkeypatch):
    cfg = replace(setting, chord_history_correction=True, chord_history_updates=True,
                  chord_attnres_backend=backend, attnres_block_size=size, depth=5)
    model = make(cfg)
    ids, _ = inputs()
    x = model.wte(ids)
    states_in = []
    initialize = model.initial_state

    def capture_start(*args):
        h, v = initialize(*args)
        states_in.append(h)
        return h, v

    monkeypatch.setattr(model, 'initial_state', capture_start)
    banks = []

    def capture(module, args):
        history, packed, shape = args
        bank = tuple(v.reshape(shape) for v in history.unbind(1)) if packed else history
        banks.append(bank)

    for layer in model.depth_reads:
        for reader in layer:
            reader.register_forward_pre_hook(capture)
    _, states = model.backbone(x, token_ids=ids, return_layers=True)
    # Check every attention and MLP read, in particular the first and the
    # incomplete final block. The candidate under construction is never visible.
    group = size // 2
    expected = []
    start, = states_in
    for i in range(cfg.depth):
        if i >= group:
            for bank in banks[2*(i-group):2*(i-group)+2]:
                assert len(bank) == len(expected)
                for val, delta in zip(bank, expected, strict=True):
                    assert torch.equal(val, delta)
        if (i+1) % group == 0 and i+1 < cfg.depth:
            end = states[:, :, i]
            expected.append(end - start)
            start = end
    assert len(banks) == 2 * (cfg.depth-group)
    # A data-only comparison misses a detach on either side of the subtraction.
    dy = torch.randn_like(expected[0])
    actual = torch.autograd.grad(banks[0][0], x, dy, retain_graph=True)[0]
    ref = torch.autograd.grad(expected[0], x, dy)[0]
    torch.testing.assert_close(actual, ref, atol=.016, rtol=.016)
