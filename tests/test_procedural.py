"""Regression tests for native-ID packing, shifted masks and full-head gradients."""
import numpy as np
import pytest

from nanochat.procedural import DYCK, TASKS, ProceduralData


@pytest.mark.parametrize('task', TASKS)
def test_packed_examples_and_resume(task):
    data = ProceduralData(task, 42, 32768, 2048, 128 if task in DYCK else 64)
    xs, ys, masks = data.batch(17, 3)
    for chunk, row in zip(data.batch(19, 1), (xs, ys, masks)):
        np.testing.assert_array_equal(chunk[0], row[2])
    pad, sep = 32767, 32766
    for x, y, mask in zip(xs, ys, masks):
        assert np.all((x >= 0) & (x < 32768))
        assert np.all(mask[x == pad] == 0)
        np.testing.assert_array_equal(y[:-1][mask[:-1]], x[1:][mask[:-1]])
        assert not mask[-1]
        ends = np.nonzero((x != pad) & np.r_[x[1:] == pad, True])[0]
        start = 0
        for end in ends:
            doc = x[start:end+1].tolist()
            if task in DYCK:
                first = 1
                assert len(doc) == data.length
            else:
                split = len(doc) - doc[::-1].index(sep)
                prompt, answer, first = doc[:split-1], doc[split:], split
                if task == 'identity':
                    expected = prompt
                elif task == 'sort':
                    expected = sorted(prompt)
                elif task == 'reverse':
                    expected = prompt[::-1]
                elif task in ('union', 'set'):
                    expected = list(dict.fromkeys(t for t in prompt if t != sep))
                elif task == 'delete':
                    expected, key = prompt[:-2], prompt[-1]
                    assert prompt[-2] == sep
                    if key in expected:
                        expected.remove(key)
                elif task == 'stack':
                    stack = []
                    for token in prompt:
                        if token:
                            assert token not in stack
                            stack.append(token)
                        else:
                            stack.pop()
                    expected = stack[::-1]
                assert answer == expected
            np.testing.assert_array_equal(np.flatnonzero(mask[start:end+1]), np.arange(first-1, len(doc)-1))
            start = end + 2
        assert len(ends) > 1 and mask.sum() > 0


def test_masked_cce_gradients_and_microbatch():
    import torch
    from nanochat.connection_experiment import configure_attention
    from nanochat.connection_model import ConnectionConfig, build_connection_model
    from nanochat.packed_training import seed_all, train_step

    configure_attention('fa2')
    seed_all(42)
    config = ConnectionConfig(variant='baseline', depth=2, width=128, heads=1, sequence_length=128, vocab_size=256)
    model = build_connection_model(config).prepare()
    opt = torch.optim.SGD(model.parameters(), lr=0.)
    data = ProceduralData('stack', 42, 256, 128, 16)
    batch = data.batch(0, 4)
    inputs, targets, mask = (torch.from_numpy(v).cuda() for v in batch)
    dense_losses = model(inputs, targets, dense=True, loss_reduction='none')
    reference = dense_losses[mask].mean()
    reference.backward()
    gradients = {k: p.grad.float().clone() for k, p in model.named_parameters()}
    result = train_step(model, opt, None, 0, microbatch=2, batch_tokens=512, masked_batch=batch)
    assert abs(reference.item() - result) < .005
    for k, p in model.named_parameters():
        diff = (gradients[k] - p.grad.float()).norm()
        assert diff <= .035 * gradients[k].norm() + 1e-6, k
