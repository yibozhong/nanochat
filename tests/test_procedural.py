"""Regression tests for native-ID packing, shifted masks and full-head gradients."""
import numpy as np
import pytest

from nanochat.procedural import DYCK, TASKS, TRANSFORMS, ProceduralData


@pytest.mark.parametrize('task', TASKS)
def test_packed_examples_and_resume(task):
    data = ProceduralData(task, 42, 32768, 2048, 128 if task in DYCK else 64)
    xs, ys, masks = data.batch(17, 3)
    for chunk, row in zip(data.batch(19, 1), (xs, ys, masks)):
        np.testing.assert_array_equal(chunk[0], row[2])
    dev = data.batch(17, 3, split='dev')
    assert not np.array_equal(dev[0], xs)
    for first, again in zip(dev, data.batch(17, 3, split='dev')):
        np.testing.assert_array_equal(first, again)
    if task in TRANSFORMS:
        for _, labels, supervised in ((xs, ys, masks), dev):
            assert set(np.unique(labels[supervised])) <= set(range(1, 100))
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


def test_union_packed_stream_uses_reference_ids_and_keeps_legacy_resume():
    fixed = ProceduralData('union', 42, 32768, 2048, 64)
    legacy = ProceduralData('union', 42, 32768, 2048, 64, 'jiang-packed-v1')
    for split in ('train', 'dev'):
        x, y, mask = fixed.batch(23, 1, split)
        old_x, old_y, old_mask = legacy.batch(23, 1, split)
        np.testing.assert_array_equal(mask, old_mask)
        assert np.any(old_y[old_mask] > 99)
        rng = np.random.RandomState([42, 23, 1] if split == 'dev' else [42, 23])
        start = 0
        for _ in range(3):
            a, b = rng.randint(1, 100, (2, 32)).tolist()
            answer = list(dict.fromkeys(a + b))
            example = a + [32766] + b + [32766] + answer
            end = start + len(example)
            np.testing.assert_array_equal(x[0, start:end], example)
            np.testing.assert_array_equal(y[0, start:end][mask[0, start:end]], answer)
            start = end + 1
        assert not np.array_equal(old_x, x)


def test_masked_cce_gradients_and_microbatch():
    import torch
    from nanochat.connection_experiment import configure_attention
    from nanochat.connection_model import ConnectionConfig, build_connection_model
    from nanochat.packed_training import seed_all, train_step, evaluate_procedural

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
    val_ce, val_count, accuracy = evaluate_procedural(model, batch, microbatch=2, with_accuracy=True)
    with torch.no_grad():
        correct = (model(inputs).argmax(-1)[mask] == targets[mask]).sum().item()
    assert accuracy == correct / int(mask.sum())
    assert evaluate_procedural(model, batch, microbatch=2) == (val_ce, val_count)
    assert val_count == int(mask.sum()) and abs(reference.item() - val_ce) < .005
    reference.backward()
    gradients = {k: p.grad.float().clone() for k, p in model.named_parameters()}
    result = train_step(model, opt, None, 0, microbatch=2, batch_tokens=512, masked_batch=batch)
    assert abs(reference.item() - result) < .005
    for k, p in model.named_parameters():
        diff = (gradients[k] - p.grad.float()).norm()
        assert diff <= .035 * gradients[k].norm() + 1e-6, k
