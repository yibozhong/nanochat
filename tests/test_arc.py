"""Arc on analytical objectives, including the actual packed-training entry point."""
import copy
import math
from types import SimpleNamespace

import pytest
import torch

from nanochat.arc import ArcAdamW
from nanochat.optim import MuonAdamW
from nanochat.packed_training import train_step


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Arc CUDA checks')


def matrix(x):
    return torch.nn.Parameter(torch.tensor(x, device='cuda', dtype=torch.float32))


def optimizer(p, mu=1., extras=()):
    return ArcAdamW([dict(kind='muon', params=[p], lr=.17), *extras], mu=mu, beta=.5)


def normalized(x):
    return x / x.norm()


def test_query_gradient_base_write_transport_and_resume():
    p = matrix([[1., -2.], [.4, 1.5]])
    w = p.detach().double().clone()
    r, v = w.norm(), torch.zeros_like(w)
    target = torch.tensor([[2., .6], [-.3, 1.]], device='cuda', dtype=torch.float64)
    opt = optimizer(p, mu=2.)
    for i in range(6):
        q = r * normalized(w + 2 * v)
        opt.zero_grad(set_to_none=True)
        with opt.gradient_point():
            torch.testing.assert_close(p.double(), q, rtol=2e-6, atol=2e-6)
            loss = .5 * (p - target.float()).square().sum()
            loss.backward()
        torch.testing.assert_close(p.double(), w, rtol=2e-6, atol=2e-6)
        g = q - target
        torch.testing.assert_close(p.grad.double(), g, rtol=2e-6, atol=2e-6)
        new = r * normalized(w - .17 * math.sqrt(2) * normalized(g))
        # Independent parallel transport formula for a tangent v.
        carried = v - ((v * new).sum() / (r*r + (w * new).sum())) * (w + new)
        obs = new - w
        obs -= new * ((new * obs).sum() / r.square())
        v = (carried + obs) / 2
        w = new
        opt.step()
        torch.testing.assert_close(p.double(), w, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(opt.state[p]['velocity'].double(), v, rtol=2e-6, atol=2e-6)
        if i == 2:
            replica = torch.nn.Parameter(p.detach().clone())
            restored = optimizer(replica, mu=2.)
            restored.load_state_dict(copy.deepcopy(opt.state_dict()))
        if i >= 3:
            restored.zero_grad(set_to_none=True)
            with restored.gradient_point():
                (.5 * (replica - target.float()).square().sum()).backward()
            restored.step()
            assert torch.equal(replica, p)
            assert torch.equal(restored.state[replica]['velocity'], opt.state[p]['velocity'])


def test_zero_mu_matches_hyperball_and_zero_gradient():
    p = matrix([[3., 4.]])
    opt = optimizer(p, mu=0.)
    for grad in (torch.tensor([[1., 2.]], device='cuda'), torch.zeros_like(p)):
        w = p.detach().clone()
        with opt.gradient_point():
            assert torch.equal(w, p)
            p.grad = grad
        opt.step()
        if grad.norm():
            torch.testing.assert_close(p, w.norm() * normalized(w - .17 * normalized(grad)))
        else:
            assert torch.equal(p, w)


def test_failed_query_restores_weights_and_rejects_checkpoint():
    p = matrix([[1., 2.]])
    opt = optimizer(p)
    with opt.gradient_point():
        p.sum().backward()
    opt.step()
    base = p.detach().clone()
    with pytest.raises(ArithmeticError), opt.gradient_point():
        assert not torch.equal(base, p)
        with pytest.raises(RuntimeError, match='checkpoint'):
            opt.state_dict()
        raise ArithmeticError
    assert torch.equal(base, p)
    with pytest.raises(RuntimeError, match='gradient'):
        opt.step()
    with pytest.raises(ValueError, match='nonzero'):
        optimizer(matrix([[0., 0.]]))


def test_accumulation_and_adamw_at_common_query():
    class Model(torch.nn.Module):
        config = SimpleNamespace(sequence_length=1, latent_feedback=False)

        def __init__(self):
            super().__init__()
            self.p = matrix([[1., 2.]])
            self.bias = matrix([.3])
            self.reads = []

        def forward(self, x, y):
            self.reads.append(self.p.detach().clone())
            return ((x @ self.p.T + self.bias - y).square() / 2).mean()

    class Data:
        def batch(self, split, start, n):
            x = torch.tensor([[1., 2.], [2., -1.]], device='cuda')
            y = torch.tensor([[2.], [.5]], device='cuda')
            return x[start % 2:start % 2 + n], y[start % 2:start % 2 + n]

    model, data = Model(), Data()
    adam = dict(kind='adamw', lr=.01, betas=(.8, .96), eps=1e-10, weight_decay=.01)
    bias = torch.nn.Parameter(model.bias.detach().clone())
    reference_adam = MuonAdamW([dict(adam, params=[bias])])
    opt = optimizer(model.p, extras=[dict(adam, params=[model.bias])])
    for step in range(2):
        base = model.p.detach().clone()
        train_step(model, opt, data, step, microbatch=1, batch_tokens=2)
        q, other_q = model.reads[-2:]
        assert torch.equal(q, other_q) and torch.equal(q, base) == (step == 0)
        x, y = data.batch('train', 0, 2)
        error = x @ q.T + bias.detach() - y
        torch.testing.assert_close(model.p.grad, error.T @ x / 2)
        bias.grad = error.mean().reshape_as(bias)
        reference_adam.step()
        torch.testing.assert_close(model.bias, bias)
