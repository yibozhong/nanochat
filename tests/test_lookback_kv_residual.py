"""Memory residual: cached current KV, prior-hidden Q, additive delta KV."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from nanochat.connection_model import ConnectionConfig, build_connection_model
from nanochat.lookback import MemoryBank
from nanochat.lookback_cost import model_cost
import test_lookback as checks


def record(name,data):
    root=Path('results/kv-lookback-checks');root.mkdir(parents=True,exist_ok=True)
    (root/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')


CONTRACTS=(
    'test_fp64_full_model_and_all_parameter_gradients_match_independent_oracle',
    'test_mask_and_arbitrary_chunk_counts',
    'test_once_only_replacement_and_shared_history_across_layers',
    'test_update_only_changes_third_and_later_chunk_and_outer_gradients',
    'test_backbone_initialization_config_and_optimizer_ownership',
    'test_cce_dense_and_compiled_checkpoint_gradients',
    'test_production_16k_2k_chunks_causality_batch_isolation_and_reset',
    'test_exact_optimizer_rng_resume',
)


@pytest.mark.parametrize('name',CONTRACTS)
def test_memory_residual_contracts(name,tmp_path,monkeypatch):
    original=checks.build
    def build(**kwargs):
        kwargs['lookback_update']='kv_residual'
        return original(**kwargs)
    monkeypatch.setattr(checks,'build',build)
    monkeypatch.setattr(checks,'record',record)
    fn=getattr(checks,name)
    if name.endswith('shared_history_across_layers'):fn(monkeypatch)
    elif name.endswith('optimizer_rng_resume'):fn(tmp_path,monkeypatch)
    else:fn()


def small(enabled=True):
    return checks.build(device='cpu',dtype=torch.float64,enabled=enabled,
        lookback_update='kv_residual',width=16,heads=2,sequence_length=16,chunk_size=4,vocab_size=32)


def test_current_kv_computed_once_and_reused_before_previous_update(monkeypatch):
    m=small();checks.nonzero_branches(m);events=[];reads=[];banks=[]
    project=m.project_memory;lookback=m.lookback.forward;revise=m.revise_and_append
    def projected(h,c,s):
        k,v=project(h,c,s);events.append(('project',k,v));return k,v
    def update(previous,k,v,c,s):
        assert events[-1][0]=='project'
        assert k is events[-1][1] and v is events[-1][2]
        reads.append((k,v));events.append(('read',k,v))
        return lookback(previous,k,v,c,s)
    def append(bank,h):
        result=revise(bank,h);banks.append(result);return result
    monkeypatch.setattr(m,'project_memory',projected)
    monkeypatch.setattr(m.lookback,'forward',update)
    monkeypatch.setattr(m,'revise_and_append',append)
    with torch.no_grad():m.hidden(torch.arange(16).view(1,16))
    assert [e[0] for e in events]==['project']+['project','read','project']*3
    for (k,v),bank in zip(reads,banks[1:]):
        assert k is bank.keys[-1] and v is bank.values[-1]
    assert not hasattr(m.lookback,'c_k') and not hasattr(m.lookback,'c_v')
    record('reuse',dict(chunks=4,lookbacks=3,shared_projection_calls=7,current_kv_reused_by_identity=True))


def test_zero_branch_preserves_both_memory_tensors_and_predictions():
    m=small();without=small(enabled=False)
    without.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith('lookback.')})
    # Backbone is nonzero, so identical predictions cannot be explained by a
    # dormant backbone. Only the memory-update branch stays zero initialized.
    with torch.no_grad():
        for b in m.blocks:
            b.attn.c_proj.weight.normal_(std=.1);b.mlp.c_proj.weight.normal_(std=.1)
        without.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith('lookback.')})
        h=torch.randn(2,16,16,dtype=torch.float64)
        actual,bank=m.hidden_from_embeddings(h,return_bank=True)
        expected,other=without.hidden_from_embeddings(h,return_bank=True)
        assert torch.equal(actual,expected)
        assert all(torch.equal(a,b) for a,b in zip(bank.keys+bank.values,other.keys+other.values))
    record('zero-init',dict(exact_identity_memory=True,exact_identity_predictions=True))


def test_delta_is_projected_branch_and_addition_is_in_memory_space():
    m=small();checks.nonzero_branches(m)
    h1=torch.randn(2,4,16,dtype=torch.float64)
    h2=torch.randn(2,4,16,dtype=torch.float64)
    first=m.revise_and_append(MemoryBank(),h1)
    k1,v1=first.keys[0],first.values[0]
    k2,v2=m.project_memory(h2,m.cos[:,4:8],m.sin[:,4:8])
    branch=m.lookback(h1,k2,v2,m.cos[:,:4],m.sin[:,:4])
    dk,dv=m.project_memory(branch,m.cos[:,:4],m.sin[:,:4])
    result=m.revise_and_append(first,h2)
    torch.testing.assert_close(result.keys[0],k1+dk,atol=0,rtol=0)
    torch.testing.assert_close(result.values[0],v1+dv,atol=0,rtol=0)
    # No final normalization and no additional H1 residual inside the delta.
    wrong=m.project_memory(h1+branch,m.cos[:,:4],m.sin[:,:4])
    assert not torch.allclose(result.keys[0],k1+wrong[0])
    assert torch.equal(result.keys[1],k2) and torch.equal(result.values[1],v2)
    assert first.keys[0] is k1 and first.values[0] is v1


def test_old_checkpoints_keep_hidden_update_and_new_mode_is_explicit():
    old=asdict(ConnectionConfig(variant='lookback',width=16,heads=2,depth=2,vocab_size=32,sequence_length=16,chunk_size=4))
    old.pop('lookback_update')
    restored=ConnectionConfig.from_checkpoint(old)
    assert restored.lookback_update=='hidden'
    old_model=build_connection_model(restored)
    restored_model=build_connection_model(ConnectionConfig.from_checkpoint(asdict(restored)))
    restored_model.load_state_dict(old_model.state_dict(),strict=True)
    new_model=small()
    with pytest.raises(RuntimeError):new_model.load_state_dict(old_model.state_dict(),strict=True)
    with pytest.raises(ValueError):ConnectionConfig(lookback_update='unknown')


def test_kv_residual_parameter_and_flops_accounting():
    old=ConnectionConfig(variant='lookback',sequence_length=16384)
    new=ConnectionConfig(variant='lookback',sequence_length=16384,lookback_update='kv_residual')
    a,b=model_cost(old,batch=2),model_cost(new,batch=2)
    assert a['components_forward']['memory_projection']==b['components_forward']['memory_projection']
    assert a['components_forward']['attention']==b['components_forward']['attention']
    assert a['estimated_forward_flops']-b['estimated_forward_flops']==7*4*2*2048*512*512
    m=small();assert sum(p.numel() for p in m.lookback.parameters())==2*16*16
    record('cost',dict(hidden=a,kv_residual=b,production_parameters=60_293_120-2*512*512))
