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


@pytest.mark.parametrize('shape', [(16, 32), (32, 16)])
def test_muon_direction_matches_native_write_and_restores_moments(shape):
    from nanochat.connection_model import schedule

    torch.manual_seed(42)
    params = [torch.nn.Parameter(torch.randn(shape, device='cuda')) for _ in range(2)]
    refs = [torch.nn.Parameter(torch.zeros_like(p)) for p in params]
    targets = [torch.randn_like(p) for p in params]

    def groups(weights):
        return [dict(kind='muon', params=weights, lr=.03, initial_lr=.03, weight_decay=0.,
                     ns_steps=5, momentum=.95, beta2=.9)]

    def build(weights):
        opt = ArcAdamW(groups(weights), direction='muon')
        opt.param_groups[0]['initial_lr'] = .03
        return opt

    opt, native = build(params), MuonAdamW(groups(refs))
    for step in (39, 40, 401, 402):
        for o in (native, opt):
            schedule(o, step, 305176, 0.)
        assert opt.param_groups[0]['momentum'] == native.param_groups[0]['momentum']
        base = [p.detach().clone() for p in params]
        historical = [opt.state[p]['velocity'].clone() for p in params]
        opt.zero_grad(set_to_none=True)
        with opt.gradient_point():
            for p, w, v, target in zip(params, base, historical, targets):
                torch.testing.assert_close(p, normalized(w+v)*w.norm(), rtol=2e-5, atol=2e-6)
                (.5 * (p-target).square().sum()).backward()
        for ref, p in zip(refs, params):
            with torch.no_grad():
                ref.zero_()
            ref.grad = p.grad.clone()
        native.step()
        opt.step()
        for p, w, v, ref in zip(params, base, historical, refs):
            # A native, zero-decay Muon write to zero gives the signed direction.
            trial = w + .03 * math.sqrt(shape[0]) * normalized(ref.detach())
            expected = opt.state[p]['radius'] * normalized(trial)
            torch.testing.assert_close(p.detach()-w, expected-w, rtol=.01, atol=3e-5)
            carried = v - (v*expected).sum() * (w+expected) / (w.square().sum() + (w*expected).sum())
            velocity = .5*(carried + expected-w)
            velocity -= expected * ((velocity*expected).sum()/expected.square().sum())
            torch.testing.assert_close(opt.state[p]['velocity'], velocity, rtol=.01, atol=3e-5)
        if step == 40:
            replicas = [torch.nn.Parameter(p.detach().clone()) for p in params]
            restored = build(replicas)
            restored.load_state_dict(copy.deepcopy(opt.state_dict()))
        if step > 40:
            schedule(restored, step, 305176, 0.)
            restored.zero_grad(set_to_none=True)
            with restored.gradient_point():
                for p, target in zip(replicas, targets):
                    (.5 * (p-target).square().sum()).backward()
            restored.step()
            for a, b in zip(replicas, params):
                assert torch.equal(a, b)
                for key, value in opt.state[b].items():
                    assert torch.equal(restored.state[a][key], value)

    untouched = params[0].detach().clone()
    saved_state = {key: value.clone() for key, value in opt.state[params[0]].items()}
    opt.zero_grad(set_to_none=True)
    with opt.gradient_point():
        params[1].sum().backward()
    opt.step()
    assert torch.equal(untouched, params[0])
    for key in ('radius', 'velocity'):
        assert torch.equal(opt.state[params[0]][key], saved_state[key])
    for key in ('momentum_buffer', 'second_momentum_buffer'):
        assert torch.equal(opt.state[params[0]][key][0], saved_state[key][0])
