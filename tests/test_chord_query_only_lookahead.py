"""At each attention site only the query projection sees the lookahead."""
from dataclasses import replace

import pytest
import torch

from nanochat.connection_model import ConnectionConfig
from nanochat.gpt import norm
from test_chord_fused_branch import test_fused_model as check_fused_model
from test_momentum import gpu, inputs, make, test_exact_optimizer_checkpoint_next_update


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_lookahead', velocity_init='token',
                            residual_projection_init='normal', lookahead_mu_init=.4,
                            ngpt_velocity_mode='lookahead_split_learned_beta', ngpt_force_rms_floor=1e-6,
                            chord_velocity_geometry='project_only', fuse_chord_read_norm=True,
                            chord_query_only_lookahead=True)


@pytest.mark.parametrize('index', [0, 1])
def test_projection_inputs_and_velocity_gradients(setting, index):
    model = make(setting)
    ids, _ = inputs()
    block = model.blocks[index]
    cos, sin = model.cos[:, :ids.size(1)], model.sin[:, :ids.size(1)]
    h, v = model.initial_state(model.wte(ids), ids)
    if index:
        h, v = model.blocks[0].momentum_forward(h, v, cos, sin)
    # Hold the incoming states independent to isolate this site's read paths.
    h, v = h.detach().requires_grad_(), v.detach().requires_grad_()
    reads, writes = {}, []
    def capture(name):
        def record(module, args):
            reads[name] = args[0]
        return record
    modules = dict(q=block.attn.c_q, k=block.attn.c_k, v=block.attn.c_v, ffn=block.mlp.c_fc)
    handles = [m.register_forward_pre_hook(capture(k)) for k,m in modules.items()]
    handles.append(block.attn_residual.register_forward_hook(lambda mod, args, result: writes.append(result)))
    try:
        block.momentum_forward(h, v, cos, sin)
    finally:
        for handle in handles:
            handle.remove()
    assert reads['k'] is reads['v']
    torch.testing.assert_close(reads['k'], norm(h.to(cos.dtype)), rtol=0, atol=0)
    preview = block.attn_residual.branch_input(h, v, cos.dtype, True)
    torch.testing.assert_close(reads['q'], preview, rtol=0, atol=0)
    assert not torch.equal(reads['q'], reads['k'])
    torch.testing.assert_close(reads['ffn'], block.mlp_residual.branch_input(*writes[0], cos.dtype, True),
                               rtol=0, atol=0)
    dv, = torch.autograd.grad(reads['k'].sum(), v, allow_unused=True)
    assert dv is None
    dv, dmu = torch.autograd.grad((reads['q'] * torch.randn_like(preview)).sum(), (v, block.attn_residual.mu))
    assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in (dv, dmu))


def test_zero_mu_matches_scalar_chord(setting):
    config = replace(setting, lookahead_mu_init=0.)
    target, control = make(config), make(replace(config, chord_query_only_lookahead=False))
    control.load_state_dict(target.state_dict(), strict=True)
    ids, labels = inputs()
    with torch.no_grad():
        ref, actual = control(ids), target(ids)
        assert (ref-actual).norm() <= .002*ref.norm()
        torch.testing.assert_close(target(ids, labels), control(ids, labels), atol=1e-4, rtol=0)


def test_compiled_gradient_causal_cache(setting):
    check_fused_model(setting, False)
