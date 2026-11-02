"""Independent attention oracle, availability causality, state and training checks."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for
from nanochat.connection_experiment import configure_attention
from nanochat.lookback import MemoryBank, chunk_block, shared_memory_mask
from nanochat.lookback_cost import model_cost
from nanochat.packed_training import seed_all
import test_connection_training_resume as resume_checks


def record(name, data):
    p=Path('results/lookback-checks');p.mkdir(parents=True,exist_ok=True)
    (p/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')


def relative(a,b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-15)).item()


def build(device='cuda', dtype=torch.bfloat16, enabled=True, **kwargs):
    seed_all(42)
    options=dict(depth=2,width=128,heads=2,sequence_length=384,vocab_size=256,chunk_size=128)
    options.update(kwargs)
    m=build_connection_model(ConnectionConfig(variant='lookback',lookback_enabled=enabled,**options))
    if dtype==torch.float64:m=m.double()
    return m.prepare(device=device,dtype=dtype)


def nonzero_branches(m):
    with torch.no_grad():
        for b in m.blocks:
            b.attn.c_proj.weight.normal_(std=.01);b.mlp.c_proj.weight.normal_(std=.01)
        if m.lookback is not None:m.lookback.c_proj.weight.normal_(std=.01)


def reference_logits(m, idx):
    """Explicit dense softmax and independent list-based revision oracle."""
    def rms(x):return x*(x.square().mean(-1,keepdim=True)+torch.finfo(x.dtype).eps).rsqrt()
    def linear(x,p):return x@p.weight.T
    def project(x,p,start,rotary=False):
        y=linear(x,p).reshape(*x.shape[:2],m.config.heads,-1)
        if rotary:
            left,right=y.chunk(2,dim=-1)
            cos=m.cos[:,start:start+x.shape[1]];sin=m.sin[:,start:start+x.shape[1]]
            y=1.2*rms(torch.cat((left*cos+right*sin,-left*sin+right*cos),dim=-1))
        return y
    def attention(q,k,v,mask=None):
        logits=torch.einsum('bqhd,bkhd->bhqk',q,k)/(q.shape[-1]**.5)
        if mask is not None:logits=logits.masked_fill(~mask,float('-inf'))
        return torch.einsum('bhqk,bkhd->bqhd',logits.softmax(-1),v).reshape(q.shape[0],q.shape[1],-1)
    def memory(h,start):
        return project(rms(h),m.memory_k,start,True),project(rms(h),m.memory_v,start)
    embeddings=rms(m.wte(idx)); bank=[]; previous=None; outputs=[]
    for start in range(0,idx.shape[1],m.config.chunk_size):
        end=min(start+m.config.chunk_size,idx.shape[1]);h=embeddings[:,start:end]
        for block in m.blocks:
            normed=rms(h)
            q=project(normed,block.attn.c_q,start,True)
            k=project(normed,block.attn.c_k,start,True);v=project(normed,block.attn.c_v,start)
            if bank:
                k=torch.cat([x[0] for x in bank]+[k],dim=1)
                v=torch.cat([x[1] for x in bank]+[v],dim=1)
            mask=torch.tensor([[j<=start+i for j in range(end)] for i in range(end-start)],device=h.device)
            h=h+linear(attention(q,k,v,mask),block.attn.c_proj)
            h=h+linear(F.relu(linear(rms(h),block.mlp.c_fc)).square(),block.mlp.c_proj)
        outputs.append(rms(h))
        if previous is not None and m.lookback is not None:
            old,old_start=previous;look=m.lookback
            q=project(rms(old),look.c_q,old_start,True)
            k=project(rms(h),look.c_k,start,True);v=project(rms(h),look.c_v,start)
            updated=old+linear(attention(q,k,v),look.c_proj)
            bank[-1]=memory(updated,old_start)
        bank.append(memory(h,start));previous=(h,start)
    hidden=torch.cat(outputs,dim=1)
    return 15*torch.tanh(linear(hidden,m.lm_head)/15)


def test_fp64_full_model_and_all_parameter_gradients_match_independent_oracle():
    m=build(device='cpu',dtype=torch.float64,width=16,heads=2,sequence_length=15,chunk_size=4,vocab_size=32)
    nonzero_branches(m);x=torch.randint(0,32,(2,15));y=(x+1)%32
    logits=15*torch.tanh(F.linear(m.hidden(x),m.lm_head.weight)/15)
    actual=F.cross_entropy(logits.flatten(0,1),y.flatten())
    expected=F.cross_entropy(reference_logits(m,x).flatten(0,1),y.flatten())
    ga=torch.autograd.grad(actual,list(m.parameters()),retain_graph=True)
    ge=torch.autograd.grad(expected,list(m.parameters()))
    torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)
    error=relative(torch.cat([g.flatten() for g in ga]),torch.cat([g.flatten() for g in ge]))
    assert error<1e-9,error
    record('fp64-oracle',dict(loss_error=abs(actual.item()-expected.item()),gradient_relative=error,partial_last_chunk=True))


def test_mask_and_arbitrary_chunk_counts():
    mask=shared_memory_mask(4,8,'cpu')
    assert torch.all(mask[:,:8]) and torch.equal(mask[:,8:],torch.ones(4,4,dtype=torch.bool).tril())
    m=build(device='cpu',dtype=torch.float64,width=16,heads=2,sequence_length=19,chunk_size=4,vocab_size=32)
    for size in (1,4,5,8,9,19):
        h=torch.randn(1,size,16,dtype=torch.float64)
        out,bank=m.hidden_from_embeddings(h,return_bank=True)
        count=(size+3)//4
        assert out.shape==h.shape and len(bank.keys)==count and bank.revisions==count-1
        assert sum(k.shape[1] for k in bank.keys)==size and bank.positions==size
    with pytest.raises(ValueError):m(torch.zeros(1,2,dtype=torch.long),kv_cache=object())


def test_once_only_replacement_and_shared_history_across_layers(monkeypatch):
    m=build(device='cpu',dtype=torch.float64,width=16,heads=2,sequence_length=16,chunk_size=4,vocab_size=32)
    nonzero_branches(m);seen=[];banks=[]
    original=m.revise_and_append
    def capture(bank,h):
        result=original(bank,h);banks.append((bank,result));return result
    monkeypatch.setattr(m,'revise_and_append',capture)
    def spy(block,x,k,v,*args):
        seen.append((k.shape[1],k.data_ptr(),v.data_ptr()))
        return chunk_block(block,x,k,v,*args)
    m.compiled_chunk=spy
    m(torch.arange(16).view(1,16))
    for i,(old,new) in enumerate(banks):
        assert new.revisions==i
        if i:
            assert all(a is b for a,b in zip(old.keys[:-1],new.keys[:-2]))
            assert not torch.equal(old.keys[-1],new.keys[-2])
            assert old.revisions==i-1
    for start in range(0,len(seen),m.config.depth):
        assert len(set(seen[start:start+m.config.depth]))==1
    record('state',dict(chunks=4,revisions=3,same_bank_all_layers=True,functional_replacement=True))


def test_update_only_changes_third_and_later_chunk_and_outer_gradients():
    m=build(device='cpu',dtype=torch.float64,width=16,heads=2,sequence_length=12,chunk_size=4,vocab_size=32)
    other=build(device='cpu',dtype=torch.float64,enabled=False,width=16,heads=2,sequence_length=12,chunk_size=4,vocab_size=32)
    nonzero_branches(m)
    other.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith('lookback.')})
    h=torch.randn(1,12,16,dtype=torch.float64,requires_grad=True)
    a=m.hidden_from_embeddings(h);b=other.hidden_from_embeddings(h)
    torch.testing.assert_close(a[:,:8],b[:,:8],atol=0,rtol=0)
    assert not torch.equal(a[:,8:],b[:,8:])
    weight=m.lookback.c_proj.weight
    probe=torch.randn_like(a)
    early=torch.autograd.grad((a[:,:8]*probe[:,:8]).sum(),(weight,h),retain_graph=True,allow_unused=True)
    assert early[0] is None or torch.count_nonzero(early[0])==0
    assert torch.count_nonzero(early[1][:,8:])==0
    late=torch.autograd.grad((a[:,8:]*probe[:,8:]).sum(),(weight,h))
    assert late[0].norm()>0 and late[1][:,:4].norm()>0 and late[1][:,4:8].norm()>0
    record('availability',dict(first_two_chunks_unchanged=True,later_gradient_to_lookback=late[0].norm().item(),full_cross_chunk_autograd=True))


def test_backbone_initialization_config_and_optimizer_ownership():
    configure_attention('sdpa_cudnn');m=build()
    options=asdict(m.config);options['variant']='baseline';seed_all(42)
    baseline=build_connection_model(ConnectionConfig(**options)).prepare()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in baseline.state_dict().items())
    assert ConnectionConfig.from_checkpoint(asdict(m.config))==m.config
    opt,_=optimizer_for(m);owner={id(p):g for g in opt.param_groups for p in g['params']}
    assert len(owner)==len(list(m.parameters()))
    assert all(owner[id(p)]['kind']=='muon' for p in m.memory_parameters())
    before={k:v.clone() for k,v in m.state_dict().items()};x=torch.randint(0,256,(1,384),device='cuda')
    with torch.no_grad():a=m(x);b=m(x)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())


def test_cce_dense_and_compiled_checkpoint_gradients():
    configure_attention('sdpa_cudnn');torch.backends.cuda.matmul.allow_tf32=False
    m=build();nonzero_branches(m);x=torch.randint(0,256,(1,384),device='cuda');y=(x+1)%256
    results={};grads={}
    for label,dense,compiled,recompute in [('dense',True,False,False),('cce',False,False,False),
        ('compiled',False,True,False),('recomputed',False,True,True)]:
        m.zero_grad(set_to_none=True)
        if compiled and m.compiled_chunk is None:m.prepare(compile_blocks=True)
        m.config.activation_checkpoint=recompute
        loss=m(x,y,dense=dense);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        grads[label]=torch.cat([p.grad.flatten().float() for p in m.parameters()])
        results[label]=loss.item()
    errors=dict(cce_dense=relative(grads['cce'],grads['dense']),
        compiled_eager=relative(grads['compiled'],grads['cce']),recomputed=relative(grads['recomputed'],grads['compiled']))
    record('gradients',dict(losses=results,relative_errors=errors))
    assert max(results.values())-min(results.values())<3e-4
    assert errors['cce_dense']<.03 and errors['compiled_eager']<.03 and errors['recomputed']<.02,errors


def test_production_16k_2k_chunks_causality_batch_isolation_and_reset():
    configure_attention('sdpa_cudnn');m=build(depth=8,width=512,heads=4,sequence_length=16384,chunk_size=2048,vocab_size=32768)
    nonzero_branches(m);m.prepare(compile_blocks=True)
    x=torch.randint(0,32768,(2,16384),device='cuda');y=x.clone()
    y[0,5120:]=(y[0,5120:]+11)%32768
    with torch.no_grad():
        a=m.hidden(x);b=m.hidden(y);again=m.hidden(x)
    torch.testing.assert_close(a[0,:5120],b[0,:5120],atol=0,rtol=0)
    torch.testing.assert_close(a[1],b[1],atol=0,rtol=0)
    torch.testing.assert_close(a,again,atol=0,rtol=0)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    record('production-causality',dict(sequence_length=16384,chunk_size=2048,depth=8,passed=True))


def test_exact_optimizer_rng_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(resume_checks,'build',lambda ignored:build(sequence_length=128,chunk_size=32,heads=1))
    resume_checks.test_optimizer_rng_checkpoint_and_fixed_gradient_next_update('lookback',tmp_path)


def test_flops_and_memory_accounting():
    c=ConnectionConfig(variant='lookback',sequence_length=16384,chunk_size=2048)
    cost=model_cost(c,batch=2);base=model_cost(ConnectionConfig(sequence_length=16384),batch=2)
    assert cost['components_forward']['attention']==base['components_forward']['attention']
    assert cost['estimated_forward_flops']>base['estimated_forward_flops']
    assert cost['logical_memory_kv_bytes']==64*1024**2
    assert cost['logical_previous_hidden_bytes']==4*1024**2
    record('cost',dict(lookback=cost,baseline=base))
