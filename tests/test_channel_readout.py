"""Channel routing, a dense structured oracle, full CE, and model integration."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for
from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.partitioned_readout import channel_partition_hidden, vocab_partition, partitioned_cross_entropy
from nanochat.packed_training import seed_all
import test_partitioned_readout as partition_checks
import test_connection_training_resume as resume_checks

MODES = ('channel_partition', 'dense_expanded')


def record(name, value):
    folder = Path('results/channel-checks'); folder.mkdir(parents=True, exist_ok=True)
    (folder/(name+'.json')).write_text(json.dumps(value, indent=2)+'\n')


def relative(a, b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-14)).item()


def explicit_matrix(weight, ids, k, d):
    """Independent sparse V x KD oracle: enumerate original coordinates."""
    matrix = weight.new_zeros((len(weight), k*d))
    for group, tokens in enumerate(ids):
        columns = torch.tensor([stream*d+group*(d//k)+channel
                                for stream in range(k) for channel in range(d//k)], device=weight.device)
        matrix[tokens[:, None], columns[None, :]] = weight[tokens]
    return matrix


def test_fp64_channel_layout_logits_and_gradients():
    torch.manual_seed(53)
    k, d, v = 4, 8, 64
    ids, inverse = vocab_partition(v, k, 12)
    h = torch.randn(7,k,d,dtype=torch.float64,requires_grad=True)
    w = torch.randn(v,d,dtype=torch.float64,requires_grad=True)
    routed = channel_partition_hidden(h)
    for group in range(k):
        expected = torch.cat([h[:,s,group*2:(group+1)*2] for s in range(k)],-1)
        torch.testing.assert_close(routed[:,group],expected,atol=0,rtol=0)
    logits = torch.cat([F.linear(routed[:,g],w[tokens]) for g,tokens in enumerate(ids)],-1)[:,inverse]
    oracle = F.linear(h.flatten(-2),explicit_matrix(w,ids,k,d))
    target = ids[0,:7]
    loss = F.cross_entropy(15*torch.tanh(logits/15),target)
    expected = F.cross_entropy(15*torch.tanh(oracle/15),target)
    actual_grads = torch.autograd.grad(loss,(h,w),retain_graph=True)
    expected_grads = torch.autograd.grad(expected,(h,w),retain_graph=True)
    torch.testing.assert_close(logits,oracle,atol=1e-12,rtol=1e-12)
    for a,b in zip(actual_grads,expected_grads): torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    # One vocabulary group reads all four streams, only its channel slice.
    routed_grad = torch.autograd.grad(logits[:,ids[0]].sum(),h)[0]
    assert (routed_grad[:,:,:2].square().sum((0,2))>0).all()
    assert torch.count_nonzero(routed_grad[:,:,2:])==0
    # Global denominator still supervises the other channel groups.
    assert (actual_grads[0].reshape(7,k,k,2).square().sum((0,1,3))>0).all()
    record('fp64',dict(logit_max_error=(logits-oracle).abs().max().item(),
        gradient_errors=[relative(a,b) for a,b in zip(actual_grads,expected_grads)],
        all_streams_reached=True,channel_slice_isolated=True,global_denominator_reaches_all_groups=True))


def test_identical_streams_reduce_rank_but_are_not_forbidden():
    torch.manual_seed(53)
    k,d,v=4,8,64
    ids,_=vocab_partition(v,k,12)
    w=torch.randn(v,d,dtype=torch.float64)
    matrix=explicit_matrix(w,ids,k,d)
    replication=torch.eye(d,dtype=torch.float64).repeat(k,1)
    assert torch.linalg.matrix_rank(matrix).item()==k*d
    assert torch.linalg.matrix_rank(matrix@replication).item()==d
    for tokens in ids:
        assert torch.linalg.matrix_rank((matrix@replication)[tokens]).item()==d//k
    record('rank',dict(independent_state_rank=k*d,identical_stream_rank=d,
        identical_stream_rank_per_vocab_group=d//k,
        note='Pre-softcap linear map ranks; capacity penalty is not a guarantee against learned collapse'))


@pytest.mark.parametrize('readout',MODES)
@pytest.mark.parametrize('scale',[.1,1.])
@pytest.mark.parametrize('one_partition',[False,True])
def test_cce_loss_and_input_weight_gradients(readout,scale,one_partition):
    from cut_cross_entropy import linear_cross_entropy
    torch.manual_seed(123)
    n,k,d,v=256,4,128,1024
    ids,inverse=[x.cuda() for x in vocab_partition(v,k,19)]
    h=(torch.randn(n,k,d,device='cuda')*scale).bfloat16().requires_grad_()
    w=(torch.randn(v,d if readout=='channel_partition' else k*d,device='cuda')*.1).bfloat16().requires_grad_()
    targets=torch.randint(0,v//k if one_partition else v,(n,),device='cuda')
    if one_partition: targets=ids[0,targets]
    if readout=='channel_partition':
        actual=partitioned_cross_entropy(channel_partition_hidden(h),w,targets,ids,inverse)
    else:
        actual=linear_cross_entropy(h.flatten(-2),w,targets,softcap=15.,shift=0,reduction='mean',
            filter_eps=None,filter_e_grad=False,filter_c_grad=False,accum_e_fp32=True,accum_c_fp32=True)
    grads=torch.autograd.grad(actual,(h,w))
    hr=h.detach().float().requires_grad_();wr=w.detach().float().requires_grad_()
    matrix=explicit_matrix(wr,ids,k,d) if readout=='channel_partition' else wr
    full_precision_logits=F.linear(hr.flatten(-2),matrix)
    expected=F.cross_entropy(15*torch.tanh(full_precision_logits/15),targets)
    ref=torch.autograd.grad(expected,(hr,wr))
    errors=[relative(a,b) for a,b in zip(grads,ref)]
    # Pinned CCE cce_lse_forward.py:90 explicitly rounds accumulated logits
    # to the input dtype before softcap/LSE. Match that forward contract;
    # retain the independent full-FP32 oracle for BOTH gradient checks.
    rounded_reference=F.cross_entropy(15*torch.tanh(full_precision_logits.detach().to(h.dtype).float()/15),targets)
    record(f'cce-{readout}-{scale}-{one_partition}',dict(cce_loss=actual.item(),
        fp32_loss=expected.item(),bf16_logit_reference_loss=rounded_reference.item(),
        fp32_loss_error=abs((actual-expected).item()),
        matched_precision_loss_error=abs((actual-rounded_reference).item()),gradient_relative_errors=errors))
    torch.testing.assert_close(actual,rounded_reference,atol=3e-5,rtol=3e-5)
    assert max(errors)<.012,errors
    assert (grads[0].float().square().sum((0,2))>0).all()


def build(readout,**kwargs):
    seed_all(42)
    model=ConnectionModel(ConnectionConfig(variant='mhc',readout=readout,**kwargs)).prepare()
    match_vanilla_initialization(model,42)
    return model


def test_shared_initialization_and_optimizer():
    models={r:build(r) for r in ('vocab_partition',*MODES)}
    reference=models['vocab_partition'].state_dict()
    counts={}
    for name,m in models.items():
        for key,value in m.state_dict().items():
            if name=='dense_expanded' and key=='lm_head.weight':continue
            assert torch.equal(value,reference[key]),key
        assert m.stream_head is None and m.partition_mixer is None
        assert all(b.attn_connection.sinkhorn and b.mlp_connection.sinkhorn for b in m.blocks)
        assert ConnectionConfig.from_checkpoint(asdict(m.config))==m.config
        counts[name]=sum(p.numel() for p in m.parameters())
    assert counts==dict(vocab_partition=59507120,channel_partition=59507120,dense_expanded=109838768)
    groups=[]
    for m in models.values():
        opt,wd=optimizer_for(m)
        groups.append(([{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups],wd))
    assert groups[0]==groups[1]==groups[2]
    record('initialization',dict(shared_parameters_bitwise_equal=True,backbone_sinkhorn_unchanged=True,
        optimizer_hyperparameters_equal=True,parameters=counts,
        head_shapes={r:list(m.lm_head.weight.shape) for r,m in models.items()},
        expanded_head_init='Independent CPU seed43, std0.0005; all same-shaped parameters match'))


@pytest.mark.parametrize('readout',MODES)
def test_real_2k_causal_cache_compile_recompute(readout,monkeypatch):
    monkeypatch.setattr(partition_checks,'build',lambda ignored,**kw:build(readout,**kw))
    partition_checks.test_causal_cache_compile_and_checkpoint()
    record('integration-'+readout,json.loads(Path('results/partition-checks/integration.json').read_text()))


@pytest.mark.parametrize('readout',MODES)
def test_exact_optimizer_rng_resume(readout,tmp_path,monkeypatch):
    monkeypatch.setattr(resume_checks,'build',lambda ignored:build(readout,depth=2,width=128,heads=1,vocab_size=256,sequence_length=128))
    resume_checks.test_optimizer_rng_checkpoint_and_fixed_gradient_next_update('mhc',tmp_path)


def test_invalid_configurations():
    for readout in MODES:
        with pytest.raises(ValueError):ConnectionConfig(variant='baseline',readout=readout)
        with pytest.raises(ValueError):ConnectionConfig(variant='mhc',readout=readout,partition_mixing='orthogonal')
        with pytest.raises(ValueError):ConnectionConfig(variant='mhc',readout=readout,architecture='native')
