"""GPU checks for uncompressed initialization and a resumable loss warmup."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import pytest
import torch

from nanochat.connection_model import ConnectionConfig, build_connection_model
from nanochat.hca_schedule import ratio_loss_weight
from nanochat.packed_training import train_step


def config(**kwargs):
    return replace(ConnectionConfig(variant='baseline', streams=1, depth=2, width=32,
        heads=2, sequence_length=32, vocab_size=64, attention_mode='dynamic_hca',
        hca_dynamic_gradient='confidence', hca_noise=False, hca_block_size=4,
        hca_router_init='uncompressed', hca_ratio_warmup_steps=2000), **kwargs)


def make(c):
    torch.manual_seed(42)
    return build_connection_model(c).to(device='cuda', dtype=torch.float64)


def test_uncompressed_init_all_layers_train_eval_and_backbone_unchanged():
    c = config(); m = make(c)
    ref = make(replace(c, hca_router_init='default', hca_ratio_warmup_steps=0))
    for (name, value), (other, expected) in zip(m.named_parameters(), ref.named_parameters()):
        assert name == other
        if '.boundary.' not in name:
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
    ids = torch.randint(0, 64, (2, 32), device='cuda')
    for training in (True, False):
        m.train(training)
        with torch.no_grad():
            _, stats = m.hidden(ids, return_hca_stats=True)
        assert stats[1] == stats[2] == 2*2*32
        for block in m.blocks:
            assert block.attn.boundary.weight.count_nonzero() == 0
            assert block.attn.boundary.bias.item() == 2.
    # Old checkpoints have the historical initialization and no warmup.
    legacy = asdict(c)
    legacy.pop('hca_router_init'); legacy.pop('hca_ratio_warmup_steps')
    loaded = ConnectionConfig.from_checkpoint(legacy)
    assert loaded.hca_router_init == 'default' and loaded.hca_ratio_warmup_steps == 0


def test_auxiliary_value_and_router_gradients_follow_global_step():
    m = make(config()).train()
    with torch.no_grad():
        for block in m.blocks:
            block.attn.c_proj.weight.normal_(0., .02)
    ids = torch.randint(0, 64, (2, 17), device='cuda'); targets = (ids+1) % 64
    router = m.blocks[0].attn.boundary.bias
    for step, expected_weight in ((0, 0.), (1000, .5), (1999, .9995), (2000, 1.), (5000, 1.)):
        parts = m(ids, targets, dense=True, return_loss_components=True, hca_training_step=step)
        torch.testing.assert_close(parts['total'], parts['ce']+expected_weight*parts['compression_loss'], rtol=0, atol=0)
        gce, = torch.autograd.grad(parts['ce'], router, retain_graph=True)
        grate, = torch.autograd.grad(parts['compression_loss'], router, retain_graph=True)
        actual, = torch.autograd.grad(parts['total'], router)
        assert grate.abs().max() > 0
        torch.testing.assert_close(actual, gce+expected_weight*grate, rtol=1e-10, atol=1e-12)
        assert m.last_hca_metrics['ratio_loss_weight'].item() == expected_weight
    with pytest.raises(ValueError, match='completed optimizer step'):
        m(ids, targets, dense=True)
    m.eval()
    parts = m(ids, targets, dense=True, return_loss_components=True)
    torch.testing.assert_close(parts['total'], parts['ce'], rtol=0, atol=0)


def test_accumulation_and_resume_use_completed_step_not_microbatch_counter():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones((), device='cuda'))
            self.config = SimpleNamespace(sequence_length=2, attention_mode='dynamic_hca', hca_ratio_warmup_steps=2000)
            self.calls = []
        def forward(self, x, y, *, hca_training_step):
            w = ratio_loss_weight(1., 2000, hca_training_step)
            self.calls.append((hca_training_step, w))
            self.last_hca_metrics = {'ratio_loss_weight': self.weight.new_tensor(w)}
            return self.weight.square()*(1+w)
    class Data:
        def batch(self, split, start, n):
            return (torch.zeros(n, 2, device='cuda', dtype=torch.long),)*2
    m = Model(); opt = torch.optim.SGD(m.parameters(), lr=.01)
    train_step(m, opt, Data(), 1000, microbatch=1, batch_tokens=8)
    assert m.calls == [(1000, .5)]*4
    saved_model = {k:v.clone() for k,v in m.state_dict().items()}
    saved_opt = opt.state_dict()
    train_step(m, opt, Data(), 1001, microbatch=1, batch_tokens=8)
    expected = m.weight.detach().clone()
    restored = Model(); restored.load_state_dict(saved_model)
    other_opt = torch.optim.SGD(restored.parameters(), lr=.01); other_opt.load_state_dict(saved_opt)
    train_step(restored, other_opt, Data(), 1001, microbatch=1, batch_tokens=8)
    torch.testing.assert_close(restored.weight, expected, rtol=0, atol=0)
    assert restored.calls == [(1001, 1001/2000)]*4


@pytest.mark.parametrize('kwargs', [dict(hca_noise=True), dict(hca_ratio_warmup_steps=-1),
    dict(hca_ratio_warmup_steps=1.5), dict(hca_dynamic_gradient='dmc'), dict(hca_router_init='bad')])
def test_invalid_warmup_recipe_rejected(kwargs):
    with pytest.raises(ValueError): config(**kwargs)
