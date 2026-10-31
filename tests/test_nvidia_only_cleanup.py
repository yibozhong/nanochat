"""Compatibility and GPU numerical parity against immutable 699dcf2 tensors."""
from dataclasses import asdict
import json
import os
from pathlib import Path

import pytest
import torch

from nanochat.connection_experiment import configure_attention,match_vanilla_initialization
from nanochat.connection_model import ConnectionConfig,ConnectionModel
from nanochat.packed_training import seed_all


def test_checkpoint_config_migration_and_removed_options():
    for variant in ('baseline','mhc'):
        legacy=dict(variant=variant,projection_backend='official',
                    connection_backend='deepseek' if variant=='baseline' else 'nvidia')
        assert ConnectionConfig.from_checkpoint(legacy)==ConnectionConfig(variant=variant)
    for bad in [dict(variant='mhc',connection_backend='deepseek'),
                dict(variant='mhc_no_manifold'),dict(projection_backend='torch')]:
        with pytest.raises((ValueError,TypeError)):ConnectionConfig.from_checkpoint(bad)
    with pytest.raises(TypeError):ConnectionConfig(projection_backend='official')


@pytest.mark.parametrize('variant',['baseline','mhc'])
def test_699dcf2_initialization_forward_backward_parity(variant):
    path=Path(os.environ['NANOCHAT_CLEANUP_GOLDEN'])/(variant+'.pt')
    ref=torch.load(path,map_location='cpu',weights_only=False)
    assert ref['tf32_override']==os.environ.get('NVIDIA_TF32_OVERRIDE')
    configure_attention('sdpa_cudnn');torch.backends.cuda.matmul.allow_tf32=False;seed_all(42)
    c=ConnectionConfig.from_checkpoint(ref['config'])
    model=ConnectionModel(c).prepare(compile_blocks=False);match_vanilla_initialization(model,42)
    with torch.no_grad():
        for b in model.blocks:
            b.attn.c_proj.weight.normal_(std=.01);b.mlp.c_proj.weight.normal_(std=.01)
    assert model.state_dict().keys()==ref['model'].keys()
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(),ref['model'][name],rtol=0,atol=0)
    loss=model(ref['x'].cuda(),ref['y'].cuda());loss.backward()
    torch.testing.assert_close(loss.cpu(),ref['loss'],rtol=0,atol=2e-6)
    errors={}
    for name,p in model.named_parameters():
        actual=p.grad.detach().float().cpu();expected=ref['grads'][name].float()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual,expected,rtol=3e-4,atol=1e-6)
        errors[name]=(actual-expected).norm().item()/max(expected.norm().item(),1e-12)
    assert max(errors.values())<3e-4
    dest=Path('results/cleanup-checks');dest.mkdir(parents=True,exist_ok=True)
    (dest/(variant+'.json')).write_text(json.dumps(dict(initial_weights_bitwise_equal=True,
        source_commit='699dcf27bdafc82b34e3ca9ef46cb62326447b69',config=asdict(c),
        tokens=2048,loss=loss.item(),reference_loss=ref['loss'].item(),gradient_relative_errors=errors),indent=2)+'\n')
