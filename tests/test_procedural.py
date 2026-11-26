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


def test_reference_union_matches_individual_examples_and_102_class_backward():
    import torch
    from torch.nn import functional as F
    from nanochat.connection_experiment import configure_attention
    from nanochat.connection_model import ConnectionConfig, build_connection_model
    from nanochat.packed_training import evaluate_procedural, seed_all, train_step

    configure_attention('fa2')
    seed_all(42)
    config = ConnectionConfig(variant='baseline', depth=2, width=128, heads=1, sequence_length=128, vocab_size=256)
    model = build_connection_model(config).prepare()
    task = ProceduralData('union', 42, 256, 128, 16)
    ids = torch.tensor(task.output_ids(), device='cuda')
    assert task.output_ids() == [*range(100), 254, 255]
    batch = task.batch(0, 4)
    x, y, mask = (torch.from_numpy(a).cuda() for a in batch)
    count = int(mask.sum())
    correct = 0
    expected_loss = 0.
    # Evaluate each original example alone, from position zero, using full native
    # logits and then selecting the same 102 classes as the upstream GPT head.
    for b, row in enumerate(batch[0]):
        starts = np.flatnonzero((row != 255) & np.r_[True, row[:-1] == 255])
        ends = np.flatnonzero((row != 255) & np.r_[row[1:] == 255, True]) + 1
        for lo, hi in zip(starts, ends):
            select = mask[b, lo:hi]
            logits = model(x[b:b+1, lo:hi])[0, select].index_select(-1, ids)
            target = y[b, lo:hi][select]
            correct += (ids[logits.argmax(-1)] == target).sum().item()
            loss = F.cross_entropy(logits, target, reduction='sum') / count
            expected_loss += loss.detach().item()
            loss.backward()
    expected = {n: p.grad.float().clone() for n, p in model.named_parameters()}
    ce, n, accuracy = evaluate_procedural(model, batch, 2, with_accuracy=True, class_ids=ids)
    assert n == count and abs(ce - expected_loss) < .003 and abs(accuracy-correct/n) < .015
    report = {}
    got = train_step(model, torch.optim.SGD(model.parameters(), lr=0), None, 0,
                     microbatch=2, batch_tokens=512, masked_batch=batch, class_ids=ids, gradient_stats=report)
    assert abs(got - expected_loss) < .003
    assert 0 < report['gradient_norm'] < float('inf')
    for name, p in model.named_parameters():
        diff = (p.grad.float() - expected[name]).norm()
        assert diff <= .04 * expected[name].norm() + 1e-5, name
    assert model.lm_head.weight.grad[100:254].count_nonzero().item() == 0
    assert all(getattr(b.attn, 'procedural_bias', None) is None for b in model.blocks)


def test_published_union_batch_sizes_masks_and_budget():
    from nanochat.union_reference import ReferenceUnion

    task = ProceduralData('union', 42, 32768, 2048, 64)
    sampler = ReferenceUnion(task, updates=250, budget_steps=64, rows_per_step=16)
    assert sampler.completed(64) == 250
    assert len(set(sampler.order[:16000].tolist())) == 16000
    for step in range(64):
        first, last = sampler.completed(step), sampler.completed(step + 1)
        n = last - first
        counts = [(i + 1) * 16 // n - i * 16 // n for i in range(n)]
        assert sum(counts) == 16 and min(counts) >= 4
        if step not in (0, 63):
            continue
        for update, rows in zip(range(first, last), counts):
            x, y, mask = sampler.batch(update, rows)
            answers = []
            for row, labels, supervised in zip(x, y, mask):
                starts = np.flatnonzero((row != 32767) & np.r_[True, row[:-1] == 32767])
                ends = np.flatnonzero((row != 32767) & np.r_[row[1:] == 32767, True]) + 1
                for start, end in zip(starts, ends):
                    prompt = row[start:start + 66]
                    assert np.flatnonzero(prompt == 32766).tolist() == [32, 65]
                    answer = list(dict.fromkeys(t for t in prompt if t != 32766))
                    np.testing.assert_array_equal(labels[start:end][supervised[start:end]], answer)
                    answers.append(tuple(prompt[prompt != 32766]))
            expected = [tuple(np.random.RandomState(42 + j).randint(1, 100, 64))
                        for j in sampler.order[update*64:(update+1)*64].tolist()]
            assert sorted(expected) == sorted(answers)


def test_reference_adamw_resume_keeps_fp32_master_and_clip():
    import copy
    import torch
    from nanochat.union_reference import ReferenceAdamW

    model = torch.nn.Linear(4, 2).cuda()
    model.weight.data = model.weight.data.bfloat16()
    initial = model.weight.detach().float().clone()
    opt = ReferenceAdamW(model)
    for p in model.parameters():
        p.grad = torch.full_like(p, 50.)
    opt.step()
    master = opt.low_precision[0][1]
    assert master.dtype == torch.float32 and not torch.equal(initial, master)
    assert torch.equal(model.weight, master.bfloat16())
    assert torch.stack([p.grad.float().norm() for p in opt.param_groups[0]['params']]).norm() <= 1.00001
    model2 = copy.deepcopy(model)
    opt2 = ReferenceAdamW(model2)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
    for m, o in ((model, opt), (model2, opt2)):
        o.zero_grad()
        for p in m.parameters():
            p.grad = torch.ones_like(p)
        o.step()
    for left, right in zip(model.parameters(), model2.parameters()):
        assert torch.equal(left, right)
    assert torch.equal(master, opt2.low_precision[0][1])
