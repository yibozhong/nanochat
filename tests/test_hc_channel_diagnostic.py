"""Independent synthetic oracles for frozen HC diagnostic measurements."""
import pytest
import torch
from scripts.connection_hc_channel_diagnostic import geometry, scalar_replacements, replace_scales, Observer


def test_orthogonal_identical_and_zero_stream_geometry():
    x = torch.eye(4).reshape(1, 1, 4, 4)
    g = geometry(x)
    assert g['effective_rank'].item() == pytest.approx(4)
    assert g['difference_fraction'].item() == pytest.approx(.75)
    assert g['mean_pair_cosine'].item() == 0
    identical = geometry(torch.ones_like(x))
    assert identical['effective_rank'].item() == pytest.approx(1)
    assert identical['difference_fraction'].item() == 0
    assert identical['mean_pair_cosine'].item() == 1
    zero = geometry(torch.zeros_like(x))
    assert torch.isnan(zero['effective_rank']).all() and zero['valid_pair_cosines'].item() == 0


def test_signed_h_can_expand_stream_differences():
    x = torch.eye(4).reshape(1, 1, 4, 4)
    h = 2*torch.eye(4)-torch.ones(4, 4)/4
    before, after = geometry(x), geometry(x@h)
    assert (after['difference_energy']/before['difference_energy']).item() == pytest.approx(4)


def test_calibrated_constants_remove_channel_variation_and_preserve_energy():
    s = torch.tensor([[.3, -1.], [1.2, -2.], [2.4, -.5]], dtype=torch.float64)
    w = torch.tensor([[1., 3.], [2., 2.], [7., 1.]], dtype=torch.float64)
    controls = scalar_replacements(s, w)
    c = controls['calibrated_energy']
    torch.testing.assert_close(c.std(0), torch.zeros(2, dtype=torch.float64), atol=1e-15, rtol=0)
    torch.testing.assert_close((w*c.square()).sum(0), (w*s.square()).sum(0))
    ls = controls['calibrated_ls']
    assert bool(((w*(s-ls).square()).sum(0) <= (w*(s-c).square()).sum(0)).all())


def test_rank_one_write_and_channel_scaled_write_can_differ():
    y = torch.arange(1., 9.)
    post = torch.tensor([1., 2., 3., 4.])
    write = y[:, None]*post
    assert geometry(write[None, None])['effective_rank'].item() == pytest.approx(1, abs=1e-6)
    assert geometry((write*torch.eye(8, 4))[None, None])['effective_rank'].item() > 1


def test_observer_orientation_exact_forward_and_exception_restoration():
    class Connection(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.sinkhorn = False
            self.input_scale = torch.nn.Parameter(torch.arange(1., 13.).reshape(3, 4))
        def write(self, y, x, mixes):
            post, h, _ = mixes
            return (x*self.input_scale)@h+y[..., None]*post[..., None, :]
    class Block: pass
    class Model: pass
    model = Model(); block = Block(); model.blocks = [block]
    block.attn_connection, block.mlp_connection = Connection(), Connection()
    c = block.attn_connection; original = c.input_scale.detach().clone()
    with pytest.raises(RuntimeError):
        with replace_scales(model, {'01_attn': torch.zeros_like(original)}):
            assert c.input_scale.count_nonzero() == 0
            raise RuntimeError('restore')
    torch.testing.assert_close(c.input_scale, original, atol=0, rtol=0)
    x = torch.arange(1., 25.).reshape(1, 2, 3, 4)
    y = torch.ones(1, 2, 3); post = torch.arange(1., 5.).expand(1, 2, 4)
    h = torch.arange(1., 17.).reshape(1, 1, 4, 4).expand(1, 2, 4, 4)
    expected = c.write(y, x, (post, h, None))
    with torch.no_grad():
        with Observer(model, 'geometry', stride=1).installed() as observer:
            got = c.write(y, x, (post, h, None))
        torch.testing.assert_close(got, expected, atol=0, rtol=0)
        assert observer.decomposition_relative['01_attn'] == 0
    assert 'write' not in c.__dict__ and 'write' not in block.mlp_connection.__dict__
