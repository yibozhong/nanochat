"""CUDA regression checks for staged first-order updates on the vanilla model."""
import copy

import pytest
import torch

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for
from nanochat.gpt import norm
from nanochat.optim import MuonAdamW
from nanochat.sequential_backward import sequential_cost, sequential_step


def tiny():
    configure_attention('fa2')
    torch.manual_seed(42)
    return ConnectionModel(ConnectionConfig(depth=2, width=128, heads=2, sequence_length=32,
                           vocab_size=256, streams=1, residual_projection_init='normal')).prepare()


class Rows:
    def __init__(self):
        self.rows = torch.randint(256, (16, 33), device='cuda')

    def batch(self, split, first, count):
        assert split == 'train'
        rows = self.rows[first:first+count]
        return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


def oracle(model, data, microbatch, step):
    # Independent per-stage optimizers with the hyperparameters from ordinary backprop.
    whole, _ = optimizer_for(model)
    ends = [model.lm_head.weight, model.wte.weight]
    params = [list(block.parameters()) for block in model.blocks]
    def optimizer(weights):
        ids = {id(w) for w in weights}
        groups = [dict(g, params=[p for p in g['params'] if id(p) in ids]) for g in whole.param_groups]
        return MuonAdamW([g for g in groups if g['params']])

    optimizers = [optimizer(weights) for weights in params]
    endpoints = optimizer(ends)
    records = []
    for offset in range(0, 4, microbatch):
        x, y = data.batch('train', step * 4 + offset, microbatch)
        emb = norm(model.wte(x))
        with torch.no_grad():
            h = emb
            inputs = []
            for block in model.blocks:
                inputs.append(h.detach())
                h = block(h, model.cos, model.sin)
        top = h.detach().requires_grad_()
        loss = model.readout(norm(top), y) * (microbatch / 4)
        dh, dw = torch.autograd.grad(loss, (top, ends[0]))
        ends[0].grad = dw if ends[0].grad is None else ends[0].grad + dw
        records.append([emb, inputs, dh, loss.item()])
    changed = []
    for i in reversed(range(len(params))):
        block = model.blocks[i]
        stale = []
        for _, inputs, gradient, _ in records:
            x = inputs[i].detach().requires_grad_()
            out = block(x, model.cos, model.sin)
            dx, *grads = torch.autograd.grad(out, (x, *params[i]), gradient)
            stale.append(dx)
            for p, grad in zip(params[i], grads):
                p.grad = grad if p.grad is None else p.grad + grad
        optimizers[i].step()
        for record, old in zip(records, stale):
            x = record[1][i].detach().requires_grad_()
            new = block(x, model.cos, model.sin)
            record[2], = torch.autograd.grad(new, x, record[2])
            changed.append((record[2] - old).float().norm().item())
    for emb, _, gradient, _ in records:
        emb.backward(gradient)
    endpoints.step()
    assert min(changed) > 0
    return sum(r[3] for r in records)


@pytest.mark.parametrize('microbatch,checkpoint', [(4, False), (2, False), (4, True)])
def test_updated_vjp_and_full_batch_gradient(microbatch, checkpoint):
    model = tiny()
    model.config.activation_checkpoint = checkpoint
    ref = copy.deepcopy(model)
    data = Rows()
    opt, _ = optimizer_for(model, sequential_backward=True)
    order = []
    opt.register_step_pre_hook(lambda optimizer, args, kwargs: order.append(kwargs['stage']))
    expected = oracle(ref, data, microbatch, 1)
    actual = sequential_step(model, opt, data, 1, microbatch=microbatch, batch_tokens=128)
    assert actual == pytest.approx(expected, abs=2e-5)
    assert order == [1, 0, 'endpoints']
    for p, q in zip(model.parameters(), ref.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=0.015, atol=2e-5)
        torch.testing.assert_close(p, q, rtol=0.005, atol=1e-5)
        assert p.requires_grad and p.grad.grad_fn is None


def test_optimizer_state_and_resume():
    model = tiny()
    ref = copy.deepcopy(model)
    staged, _ = optimizer_for(model, sequential_backward=True)
    regular, _ = optimizer_for(ref)
    for update in range(3):
        for p, q in zip(model.parameters(), ref.parameters(), strict=True):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        for stage in [1, 0, 'endpoints']:
            staged.step(stage=stage)
        regular.step()
        for p, q in zip(model.parameters(), ref.parameters(), strict=True):
            torch.testing.assert_close(p, q, rtol=0.003, atol=1e-4)
        for p in (model.wte.weight, model.lm_head.weight):
            assert staged.state[p]['step'] == update + 1
        if update == 0:
            restored = copy.deepcopy(model)
            restored_opt, _ = optimizer_for(restored, sequential_backward=True)
            restored_opt.load_state_dict(copy.deepcopy(staged.state_dict()))
            model, staged = restored, restored_opt


@pytest.mark.parametrize('tied', ['blocks', 'vocabulary'])
def test_shared_parameters(tied):
    model = tiny()
    if tied == 'blocks':
        model.blocks[1].mlp.c_fc.weight = model.blocks[0].mlp.c_fc.weight
    else:
        model.lm_head.weight = model.wte.weight
    with pytest.raises(ValueError, match='shared parameters'):
        optimizer_for(model, sequential_backward=True)


def test_matrix_accounting():
    c = ConnectionConfig(streams=1)
    full = sequential_cost(c, 16, 16)
    recompute = sequential_cost(c, 16, 8)
    parts = full['components_forward']
    assert full['standard_training_flops'] == 3 * sum(parts.values())
    assert full['estimated_training_flops'] > full['standard_training_flops']
    assert recompute['estimated_training_flops'] - full['estimated_training_flops'] == (
        parts['backbone_projections'] + parts['attention'])
