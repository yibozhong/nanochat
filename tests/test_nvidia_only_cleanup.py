"""Compatibility and GPU numerical parity against immutable 699dcf2 tensors."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import runpy

import pytest
import torch

from nanochat.connection_experiment import configure_attention,match_vanilla_initialization
from nanochat.connection_model import ConnectionConfig,ConnectionModel
from nanochat.packed_training import seed_all

HELPERS = runpy.run_path(str(Path(__file__).resolve().parents[1] /
    'research/connections/nvidia-only/parity_helpers.py'))


def test_checkpoint_config_migration_and_removed_options():
    for variant in ('baseline','mhc'):
        legacy=dict(variant=variant,projection_backend='official',
                    connection_backend='deepseek' if variant=='baseline' else 'nvidia')
        assert ConnectionConfig.from_checkpoint(legacy)==ConnectionConfig(variant=variant)
    for bad in [dict(variant='mhc',connection_backend='deepseek'),
                dict(variant='mhc_no_manifold'),dict(projection_backend='torch')]:
        with pytest.raises((ValueError,TypeError)):ConnectionConfig.from_checkpoint(bad)
    with pytest.raises(TypeError):ConnectionConfig(projection_backend='official')


def test_retained_checkpoint_metadata_after_chunk_experiment_removal():
    inert = dict(chunk_size=2048, lookback_enabled=True, lookback_update='hidden',
                 recompute_chunk_size=2048, recompute_previous=True,
                 recompute_input='final_hidden', recompute_memory='last_attention_kv_blend')
    for architecture in ('vanilla', 'native'):
        for variant in ('baseline', 'mhc', 'mhc_no_sinkhorn'):
            clean = ConnectionConfig(architecture=architecture, variant=variant,
                                     sequence_length=16384)
            legacy = dict(asdict(clean), **inert)
            original = dict(legacy)
            assert ConnectionConfig.from_checkpoint(legacy) == clean
            assert legacy == original  # Reading metadata must not mutate it.
    for variant in ('lookback', 'lookback_recompute'):
        with pytest.raises(ValueError, match='retired'):
            ConnectionConfig.from_checkpoint(dict(variant=variant, **inert))
        with pytest.raises(ValueError):
            ConnectionConfig(variant=variant)
    with pytest.raises(TypeError):
        ConnectionConfig(recompute_chunk_size=2048)


@pytest.mark.parametrize('variant',['baseline','mhc'])
def test_699dcf2_initialization_forward_backward_parity(variant):
    path=Path(os.environ['NANOCHAT_CLEANUP_GOLDEN'])/(variant+'.pt')
    ref=torch.load(path,map_location='cpu',weights_only=False)
    assert ref['method']==HELPERS['METHOD']
    assert ref['tf32_override']==os.environ.get('NVIDIA_TF32_OVERRIDE')
    assert ref['input_tokens_unique']==ref['x'].numel()==2048
    configure_attention('sdpa_cudnn');torch.backends.cuda.matmul.allow_tf32=False;seed_all(42)
    c=ConnectionConfig.from_checkpoint(ref['config'])
    model=ConnectionModel(c).prepare(compile_blocks=False);match_vanilla_initialization(model,42)
    with torch.no_grad():
        for b in model.blocks:
            b.attn.c_proj.weight.normal_(std=.01);b.mlp.c_proj.weight.normal_(std=.01)
    assert model.state_dict().keys()==ref['model'].keys()
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(),ref['model'][name],rtol=0,atol=0,msg=name)
    comparisons={}
    for objective in ('dense_fp32','production_cce'):
        actual=HELPERS['evaluate'](model,ref['x'].cuda(),ref['y'].cuda(),objective)
        comparisons[objective]=HELPERS['compare'](actual,ref['samples'][objective],objective)
        del actual
    dest=Path('results/cleanup-checks');dest.mkdir(parents=True,exist_ok=True)
    # Write every named parameter before asserting, including on failure.
    (dest/(variant+'.json')).write_text(json.dumps(dict(initial_weights_bitwise_equal=True,
        source_commit='699dcf27bdafc82b34e3ca9ef46cb62326447b69',config=asdict(c),
        tokens=2048,unique_input_tokens=ref['input_tokens_unique'],method=HELPERS['METHOD'],
        comparisons=comparisons,reference_repeat_control=ref['repeat_control']),indent=2)+'\n')
    failures={}
    for origin,values in (('old_old',ref['repeat_control']),('new_old',comparisons)):
        for objective,result in values.items():
            if not result['passed']:
                failures[origin+'/'+objective]={k:v for k,v in result.items() if k!='parameters'}
                failures[origin+'/'+objective]['worst_parameters']=sorted(
                    result['parameters'].items(),key=lambda p:p[1]['mixed_l2_ratio'],reverse=True)[:5]
    assert not failures,failures
