"""Independent dense oracle and availability tests for Transformer reruns."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from nanochat import flash_attention
from nanochat.chunk_recompute import memory_causal_mask, transformer_chunk, blend_previous
from nanochat.connection_cost import model_cost
from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for
from nanochat.packed_training import seed_all
import test_connection_training_resume as resume_checks


@pytest.fixture(autouse=True)
def isolated_attention_policy():
    # CUDA-only policy must not leak from another test into CPU FP64 oracles.
    names=('flash','mem_efficient','math','cudnn')
    before={n:getattr(torch.backends.cuda,n+'_sdp_enabled')() for n in names}
    fa3=flash_attention.USE_FA3
    flash_attention.USE_FA3=False
    torch.backends.cuda.enable_math_sdp(True)
    yield
    flash_attention.USE_FA3=fa3
    for n,v in before.items():getattr(torch.backends.cuda,'enable_'+n+'_sdp')(v)


def record(name,data):
    p=Path('results/recompute-checks');p.mkdir(parents=True,exist_ok=True)
    (p/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')


def relative(a,b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-15)).item()


def build(device='cpu',dtype=torch.float64,**kwargs):
    seed_all(42)
    opts=dict(variant='lookback_recompute',depth=3,width=16,heads=2,
        sequence_length=15,recompute_chunk_size=4,vocab_size=32)
    opts.update(kwargs)
    m=build_connection_model(ConnectionConfig(**opts))
    if dtype==torch.float64:m=m.double()
    return m.prepare(device=device,dtype=dtype)


def gpu_model(**kwargs):
    opts=dict(depth=2,width=128,heads=2,sequence_length=384,
        recompute_chunk_size=128,vocab_size=256);opts.update(kwargs)
    return build(device='cuda',dtype=torch.bfloat16,**opts)


def nonzero_branches(m):
    with torch.no_grad():
        for block in m.blocks:
            block.attn.c_proj.weight.normal_(std=.01)
            block.mlp.c_proj.weight.normal_(std=.01)


def reference_hidden(m,idx):
    """Explicit dense softmax; no production attention, masking or bank helpers."""
    def rms(x):return x*(x.square().mean(-1,keepdim=True)+torch.finfo(x.dtype).eps).rsqrt()
    def linear(x,p):return x@p.weight.T
    def projection(h,p,start,rotary):
        z=linear(h,p).reshape(*h.shape[:2],m.config.heads,-1)
        if rotary:
            left,right=z.chunk(2,dim=-1)
            co=m.cos[:,start:start+h.shape[1]];si=m.sin[:,start:start+h.shape[1]]
            z=1.2*rms(torch.cat((left*co+right*si,right*co-left*si),dim=-1))
        return z
    embedding=rms(m.wte(idx))
    def run(h,start,end,memories):
        saved=[]
        for layer,block in enumerate(m.blocks):
            z=rms(h);q=projection(z,block.attn.c_q,start,True)
            k=projection(z,block.attn.c_k,start,True);v=projection(z,block.attn.c_v,start,False)
            saved.append((k,v));past_k=[];past_v=[]
            for mem in memories:
                mk,mv=mem['deep']
                if mem is memories[-1] and mem['end']==start:
                    lk,lv=mem['local'][layer]
                    a=torch.sigmoid(m.prev_chunk_alphas[layer])[None,None,:,None]
                    mk=a*mk+(1-a)*lk;mv=a*mv+(1-a)*lv
                past_k.append(mk);past_v.append(mv)
            keys=torch.cat(past_k+[k],dim=1)
            values=torch.cat(past_v+[v],dim=1)
            history=keys.shape[1]-(end-start)
            mask=torch.tensor([[j<history or j-history<=i for j in range(keys.shape[1])]
                for i in range(end-start)],device=h.device)
            scores=torch.einsum('bqhd,bkhd->bhqk',q,keys)/(q.shape[-1]**.5)
            weights=scores.masked_fill(~mask,float('-inf')).softmax(-1)
            attention=torch.einsum('bhqk,bkhd->bqhd',weights,values).reshape_as(h)
            h=h+linear(attention,block.attn.c_proj)
            h=h+linear(F.relu(linear(rms(h),block.mlp.c_fc)).square(),block.mlp.c_proj)
        return h,dict(deep=saved[-1],local=saved,end=end)
    outputs=[];memories=[];positions=[];previous_hidden=None
    for start in range(0,idx.shape[1],m.config.recompute_chunk_size):
        end=min(idx.shape[1],start+m.config.recompute_chunk_size)
        output,current=run(embedding[:,start:end],start,end,memories);outputs.append(rms(output))
        if memories and m.config.recompute_previous:
            previous_start,previous_end=positions[-1]
            _,updated=run(previous_hidden,previous_start,previous_end,memories[:-1]+[current])
            memories[-1]=updated
        memories.append(current);positions.append((start,end));previous_hidden=output
    return torch.cat(outputs,dim=1)


def loss64(m,hidden,y):
    logits=15*torch.tanh(F.linear(hidden,m.lm_head.weight)/15)
    return F.cross_entropy(logits.flatten(0,1),y.flatten())


def test_fp64_independent_forward_and_all_parameter_gradient_oracle():
    m=build();nonzero_branches(m);x=torch.randint(0,32,(2,15));y=(x+1)%32
    actual=loss64(m,m.hidden(x),y);expected=loss64(m,reference_hidden(m,x),y)
    ga=torch.autograd.grad(actual,tuple(m.parameters()),retain_graph=True)
    ge=torch.autograd.grad(expected,tuple(m.parameters()))
    torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)
    error=relative(torch.cat([g.flatten() for g in ga]),torch.cat([g.flatten() for g in ge]))
    assert error<1e-9,error
    record('fp64-oracle',dict(loss_error=abs(actual.item()-expected.item()),gradient_relative=error))


def test_single_chunk_equals_full_vanilla_forward_backward():
    errors=[]
    for chunk,enabled in ((15,False),(15,True),(32,True)):
        m=build(recompute_chunk_size=chunk,recompute_previous=enabled);nonzero_branches(m)
        base=build(variant='baseline');base.load_state_dict({k:v for k,v in m.state_dict().items() if k!='prev_chunk_alphas'})
        x=torch.randint(0,32,(2,15));y=(x+1)%32
        a=m.hidden(x);b=base.hidden(x)
        torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)
        ga=torch.autograd.grad(loss64(m,a,y),tuple(p for k,p in m.named_parameters() if k!='prev_chunk_alphas'))
        gb=torch.autograd.grad(loss64(base,b,y),tuple(base.parameters()))
        error=relative(torch.cat([g.flatten() for g in ga]),torch.cat([g.flatten() for g in gb]))
        assert error<1e-9,error;errors.append(error)
    record('vanilla-equivalence',dict(gradient_relative=errors))


def test_no_lookback_shared_memory_matches_independent_oracle():
    m=build(recompute_previous=False);nonzero_branches(m)
    x=torch.randint(0,32,(2,15));y=(x+1)%32
    a=loss64(m,m.hidden(x),y);b=loss64(m,reference_hidden(m,x),y)
    ga=torch.autograd.grad(a,tuple(m.parameters()),retain_graph=True)
    gb=torch.autograd.grad(b,tuple(m.parameters()))
    torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)
    error=relative(torch.cat([g.flatten() for g in ga]),torch.cat([g.flatten() for g in gb]))
    assert error<1e-9
    record('no-lookback-oracle',dict(gradient_relative=error,uses_last_attention_kv_and_previous_blend=True))


def test_lastkv_blend_matches_closed_form_headwise_derivatives():
    tensors=[torch.randn(2,5,3,4,dtype=torch.float64,requires_grad=True) for _ in range(4)]
    logits=torch.tensor([-2.,0.,1.5],dtype=torch.float64,requires_grad=True)
    k,v=blend_previous(*tensors,logits);gk=torch.randn_like(k);gv=torch.randn_like(v)
    grads=torch.autograd.grad((k*gk).sum()+(v*gv).sum(),(*tensors,logits))
    a=(1/(1+torch.exp(-logits))).view(1,1,3,1)
    dk,dv,lk,lv=tensors
    expected=(a*gk,a*gv,(1-a)*gk,(1-a)*gv,
        (a*(1-a)*((dk-lk)*gk+(dv-lv)*gv)).sum((0,1,3)))
    for actual,reference in zip(grads,expected):torch.testing.assert_close(actual,reference,atol=1e-12,rtol=1e-12)
    assert torch.all(grads[-1].abs()>0)
    record('blend-derivatives',dict(per_head=True,same_gate_for_kv=True,all_input_and_logit_gradients=True))


def test_two_chunks_local_gate_limit_recovers_vanilla():
    m=build(sequence_length=8,recompute_previous=False);nonzero_branches(m)
    with torch.no_grad():m.prev_chunk_alphas.fill_(-100)
    base=build(variant='baseline',sequence_length=8)
    base.load_state_dict({k:v for k,v in m.state_dict().items() if k!='prev_chunk_alphas'})
    x=torch.randint(0,32,(2,8));y=(x+1)%32
    a=m.hidden(x);b=base.hidden(x)
    torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)
    ga=torch.autograd.grad(loss64(m,a,y),tuple(p for n,p in m.named_parameters() if n!='prev_chunk_alphas'))
    gb=torch.autograd.grad(loss64(base,b,y),tuple(base.parameters()))
    error=relative(torch.cat([v.flatten() for v in ga]),torch.cat([v.flatten() for v in gb]));assert error<1e-9
    record('local-gate-limit',dict(two_chunks_match_vanilla=True,gradient_relative=error))


def test_cache_is_last_attention_kv_before_final_block_output():
    m=build(sequence_length=12,recompute_previous=False);nonzero_branches(m)
    h=torch.randn(1,4,16,dtype=torch.float64)
    old,state=m.process_chunk(h,0,())
    with torch.no_grad():m.blocks[-1].mlp.c_proj.weight.add_(.1)
    new,changed=m.process_chunk(h,0,())
    assert not torch.equal(old,new)
    for a,b in zip(state.kv,changed.kv):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert state.kv is state.local_kv[-1]
    _,bank=m.hidden_from_embeddings(torch.randn(1,12,16,dtype=torch.float64),return_bank=True)
    assert all(c.local_kv is None and c.final_hidden is None for c in bank[:-1])
    assert bank[-1].local_kv is not None and bank[-1].final_hidden is None
    record('last-attention-provenance',dict(final_mlp_does_not_change_cache=True,no_extra_projection=True,
        only_latest_original_chunk_keeps_layer_kv=True))


def test_self_and_history_attention_share_one_softmax_with_local_causal_mask():
    m=build(sequence_length=4);block=m.blocks[0];d=m.config.width
    with torch.no_grad():
        block.attn.c_q.weight.zero_();block.attn.c_k.weight.zero_()
        block.attn.c_v.weight.copy_(torch.eye(d,dtype=torch.float64))
        block.attn.c_proj.weight.copy_(torch.eye(d,dtype=torch.float64))
        block.mlp.c_proj.weight.zero_()
    h=torch.randn(1,4,d,dtype=torch.float64)
    memory_k=torch.zeros(1,3,2,d//2,dtype=torch.float64)
    memory_v=torch.randn_like(memory_k)
    actual,_,_=transformer_chunk(block,h,memory_k,memory_v,m.cos,m.sin,memory_causal_mask(4,3,'cpu'))
    normalized=h*(h.square().mean(-1,keepdim=True)+torch.finfo(h.dtype).eps).rsqrt()
    expected=h+(memory_v.reshape(1,3,d).sum(1,keepdim=True)+normalized.cumsum(1))/torch.arange(4,8,dtype=h.dtype)[None,:,None]
    torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)
    future_only=h+memory_v.reshape(1,3,d).mean(1,keepdim=True)
    assert not torch.allclose(actual,future_only)
    record('self-and-history',dict(joint_softmax=True,local_causal_self_attention=True,future_only_rejected=True))


def test_once_only_schedule_final_hidden_input_and_functional_bank(monkeypatch):
    m=build(sequence_length=16);nonzero_branches(m)
    embeddings=torch.randn(1,16,16,dtype=torch.float64);calls=[];boundaries=[];original_hidden={}
    run=m.process_chunk;advance=m.advance;layer_outputs=[];layer_memory=[];original_states={}
    def layer_spy(block,x,k,v,*args):
        layer_memory.append((k.data_ptr(),v.data_ptr(),k.shape[1]))
        result=transformer_chunk(block,x,k,v,*args);layer_outputs.append(result);return result
    m.compiled_chunk=layer_spy
    def spy(hidden,start,history):
        rerun=any(c.start>start for c in history)
        if rerun:
            assert hidden is original_hidden[start]
            assert hidden.data_ptr()!=embeddings[:,start:start+hidden.shape[1]].data_ptr()
        else:
            assert hidden.data_ptr()==embeddings[:,start:start+hidden.shape[1]].data_ptr()
        calls.append((start,[c.start for c in history]))
        result=run(hidden,start,history)
        assert all(a is b for a,b in zip(result[1].kv,layer_outputs[-1][1:]))
        assert result[1].kv is result[1].local_kv[-1]
        if rerun:
            assert len(set(layer_memory[-m.config.depth:]))==1
        if not rerun:
            original_hidden[start]=result[0]
            original_states[start]=result[1]
            assert result[1].final_hidden is result[0]
        return result
    def spy_advance(*args):
        old=args[-1];out,new=advance(*args);boundaries.append((old,new));return out,new
    monkeypatch.setattr(m,'process_chunk',spy);monkeypatch.setattr(m,'advance',spy_advance)
    _,bank=m.hidden_from_embeddings(embeddings,return_bank=True)
    assert calls==[(0,[]),(4,[0]),(0,[4]),(8,[0,4]),(4,[0,8]),(12,[0,4,8]),(8,[0,4,12])]
    assert [c.revisions for c in bank]==[1,1,1,0]
    assert len(layer_outputs)==7*m.config.depth
    for old,new in boundaries[1:]:
        assert old[-1].revisions==0 and new[-2].revisions==1
        assert all(a is b for a,b in zip(old[:-1],new[:-2]))
        assert not torch.equal(old[-1].kv[0],new[-2].kv[0])
        assert not torch.equal(old[-1].kv[1],new[-2].kv[1])
        assert old[-1].final_hidden is not None and new[-2].final_hidden is None
        assert old[-1].local_kv is not None and new[-2].local_kv is None
    assert bank[-1].final_hidden is original_hidden[12]
    assert sum(c.local_kv is not None for c in bank)==1
    assert boundaries[1][1][-1].kv[0] is original_states[4].kv[0]
    assert boundaries[2][0][-1].kv[0] is original_states[4].kv[0]
    record('schedule',dict(passes=calls,functional_replacement=True,rerun_shared_kv_same_tensor_all_layers=True,
        extra_projection_calls=0,last_attention_pair_reused=True,successor_memory_reused=True,
        rerun_input='original_final_block_output_before_readout_norm',same_tensor_no_norm_no_detach=True))


def test_first_two_predictions_unchanged_and_only_later_loss_reaches_revised_cache(monkeypatch):
    m=build(sequence_length=12);nonzero_branches(m)
    other=build(sequence_length=12,recompute_previous=False);other.load_state_dict(m.state_dict())
    revised=[];run=m.process_chunk
    def spy(embedding,start,history):
        result=run(embedding,start,history)
        if any(c.start>start for c in history):revised.append(result[1].kv[1])
        return result
    monkeypatch.setattr(m,'process_chunk',spy)
    h=torch.randn(1,12,16,dtype=torch.float64,requires_grad=True)
    a=m.hidden_from_embeddings(h);b=other.hidden_from_embeddings(h)
    torch.testing.assert_close(a[:,:8],b[:,:8],atol=0,rtol=0)
    assert not torch.equal(a[:,8:],b[:,8:])
    probe=torch.randn_like(a)
    early=torch.autograd.grad((a[:,:8]*probe[:,:8]).sum(),(revised[0],h),retain_graph=True,allow_unused=True)
    assert early[0] is None or torch.count_nonzero(early[0])==0
    assert torch.count_nonzero(early[1][:,8:])==0
    late=torch.autograd.grad((a[:,8:]*probe[:,8:]).sum(),(revised[0],h))
    assert late[0].norm()>0 and late[1][:,:4].norm()>0 and late[1][:,4:8].norm()>0
    record('availability',dict(early_predictions_unchanged=True,revised_cache_gradient=late[0].norm().item()))


def test_revised_previous_can_read_last_successor_token_but_original_predictions_cannot():
    m=build(sequence_length=8);nonzero_branches(m)
    h=torch.randn(1,8,16,dtype=torch.float64,requires_grad=True)
    out,bank=m.hidden_from_embeddings(h,return_bank=True)
    probe=torch.randn_like(bank[0].kv[1][:,0])
    grad=torch.autograd.grad((bank[0].kv[1][:,0]*probe).sum(),h,retain_graph=True)[0]
    assert grad[:,-1].norm()>0
    old_grad=torch.autograd.grad(out[:,:7].square().sum(),h)[0]
    assert torch.count_nonzero(old_grad[:,7:])==0
    record('successor-visible',dict(last_successor_token_gradient=grad[:,-1].norm().item()))


def test_later_loss_backpropagates_through_original_final_hidden_into_original_stack(monkeypatch):
    m=build(sequence_length=12);nonzero_branches(m);run=m.process_chunk;finals={};revised=[];original_kv={}
    def capture(hidden,start,history):
        result=run(hidden,start,history)
        if any(c.start>start for c in history):revised.append(result[1].kv[1])
        else:finals[start]=result[0];original_kv[start]=result[1].kv[1]
        return result
    monkeypatch.setattr(m,'process_chunk',capture)
    h=torch.randn(1,12,16,dtype=torch.float64,requires_grad=True)
    out=m.hidden_from_embeddings(h)
    probe=torch.randn_like(revised[0])
    value=(revised[0]*probe).sum()
    g=torch.autograd.grad(value,(finals[0],original_kv[4],m.blocks[-1].attn.c_v.weight,h),retain_graph=True)
    assert all(v.norm()>0 for v in g) and torch.count_nonzero(g[-1][:,8:])==0
    assert torch.autograd.grad(value,finals[4],allow_unused=True,retain_graph=True)[0] is None
    late=torch.autograd.grad((out[:,8:]*torch.randn_like(out[:,8:])).sum(),revised[0])[0]
    assert late.norm()>0
    record('final-hidden-gradient',dict(original_final_hidden_gradient=g[0].norm().item(),
        successor_last_attention_v_gradient=g[1].norm().item(),last_attention_v_weight_gradient=g[2].norm().item(),
        later_loss_to_revised_shared_v=late.norm().item()))


def test_mask_arbitrary_partial_chunks_and_standard_cache_rejection():
    mask=memory_causal_mask(4,8,'cpu')
    assert mask[:,:8].all() and torch.equal(mask[:,8:],torch.ones(4,4,dtype=torch.bool).tril())
    m=build(sequence_length=19)
    for size in (1,4,5,8,9,19):
        h=torch.randn(1,size,16,dtype=torch.float64)
        out,bank=m.hidden_from_embeddings(h,return_bank=True)
        count=(size+3)//4
        assert out.shape==h.shape and len(bank)==count and sum(c.revisions for c in bank)==count-1
        assert sum(c.length for c in bank)==size
    with pytest.raises(ValueError):m(torch.zeros(1,2,dtype=torch.long),kv_cache=object())


def test_identical_parameter_initialization_rng_optimizer_and_checkpoint_schema():
    configure_attention('sdpa_cudnn');m=gpu_model();rng=torch.get_rng_state()
    baseline=gpu_model(variant='baseline')
    common={k:v for k,v in m.state_dict().items() if k!='prev_chunk_alphas'}
    # Filtering produces a plain dict; compare equivalent mappings while
    # retaining exact dtype/shape/value checks for every shared tensor.
    resume_checks.equal_tree(common,dict(baseline.state_dict()))
    assert torch.equal(rng,torch.get_rng_state())
    assert ConnectionConfig.from_checkpoint(asdict(m.config))==m.config
    a,wd=optimizer_for(m);b,wd2=optimizer_for(baseline)
    assert wd==wd2
    left=[{k:v for k,v in g.items() if k!='params'} for g in a.param_groups if not any(p is m.prev_chunk_alphas for p in g['params'])]
    right=[{k:v for k,v in g.items() if k!='params'} for g in b.param_groups]
    resume_checks.equal_tree(left,right)
    owner={id(p):g['kind'] for g in a.param_groups for p in g['params']}
    assert all(owner[id(p)]=='adamw' for p in m.memory_parameters())
    gate=next(g for g in a.param_groups if any(p is m.prev_chunk_alphas for p in g['params']))
    assert gate['weight_decay']==0 and gate['lr']==a.param_groups[0]['lr']
    assert set(m.state_dict())-set(common)=={'prev_chunk_alphas'}
    assert torch.count_nonzero(m.prev_chunk_alphas)==0
    assert sum(p.numel() for p in m.parameters())-sum(p.numel() for p in baseline.parameters())==m.config.depth*m.config.heads
    record('parameters',dict(backbone_initialization_identical=True,optimizer_hyperparameters_identical=True,
        new_projections=0,new_parameters=m.config.depth*m.config.heads,gate_optimizer='adamw_no_decay'))


def test_cce_dense_compiled_and_activation_checkpoint_gradients():
    configure_attention('sdpa_cudnn');torch.backends.cuda.matmul.allow_tf32=False
    m=gpu_model();nonzero_branches(m);x=torch.randint(0,256,(1,384),device='cuda');y=(x+1)%256
    losses={};grads={}
    for label,dense,compiled,recompute in [('dense',True,False,False),('cce',False,False,False),
        ('compiled',False,True,False),('checkpoint',False,True,True)]:
        m.zero_grad(set_to_none=True)
        if compiled and m.compiled_chunk is None:m.prepare(compile_blocks=True)
        m.config.activation_checkpoint=recompute
        loss=m(x,y,dense=dense);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        losses[label]=loss.item();grads[label]=torch.cat([p.grad.flatten().float() for p in m.parameters()])
    errors=dict(cce_dense=relative(grads['cce'],grads['dense']),compiled=relative(grads['compiled'],grads['cce']),
        checkpoint=relative(grads['checkpoint'],grads['compiled']))
    record('gradients',dict(losses=losses,relative_errors=errors))
    assert max(losses.values())-min(losses.values())<3e-4,losses
    assert errors['cce_dense']<.03 and errors['compiled']<.03 and errors['checkpoint']<.02,errors


def test_production_16k_causality_batch_isolation_and_reset():
    configure_attention('sdpa_cudnn')
    m=gpu_model(depth=8,width=512,heads=4,sequence_length=16384,recompute_chunk_size=2048,vocab_size=32768)
    nonzero_branches(m);m.prepare(compile_blocks=True)
    assert sum(p.numel() for p in m.parameters())==58720288
    x=torch.randint(0,32768,(2,16384),device='cuda');changed=x.clone()
    changed[0,5120:]=(changed[0,5120:]+11)%32768
    with torch.no_grad():a=m.hidden(x);b=m.hidden(changed);again=m.hidden(x)
    torch.testing.assert_close(a[0,:5120],b[0,:5120],atol=0,rtol=0)
    torch.testing.assert_close(a[1],b[1],atol=0,rtol=0)
    torch.testing.assert_close(a,again,atol=0,rtol=0)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    record('production-causality',dict(context=16384,chunk=2048,depth=8,passed=True,parameters=58720288))


def test_exact_optimizer_rng_checkpoint_restore(tmp_path,monkeypatch):
    monkeypatch.setattr(resume_checks,'build',lambda ignored:gpu_model(sequence_length=128,recompute_chunk_size=32,heads=1))
    resume_checks.test_optimizer_rng_checkpoint_and_fixed_gradient_next_update('lookback_recompute',tmp_path)


def test_analytical_flops_and_logical_bank_accounting():
    c=ConnectionConfig(variant='lookback_recompute',sequence_length=16384)
    cost=model_cost(c,batch=2);base=model_cost(replace(c,variant='baseline'),batch=2)
    assert cost['components_forward']['attention']==base['components_forward']['attention']
    assert cost['estimated_training_flops']>base['estimated_training_flops']
    assert cost['logical_memory_kv_bytes']==120*1024**2
    assert cost['logical_previous_layer_kv_extra_bytes']==56*1024**2
    assert cost['logical_previous_hidden_bytes']==4*1024**2
    no=model_cost(replace(c,recompute_previous=False),batch=2)
    assert base['estimated_training_flops']==no['estimated_training_flops']<cost['estimated_training_flops']
    assert no['components_forward']['lookback']==0 and no['components_forward']['memory_projection']==0
    assert no['logical_previous_hidden_bytes']==0
    record('cost',dict(recompute=cost,baseline=base))


def test_invalid_model_options():
    with pytest.raises(ValueError):ConnectionConfig(variant='lookback_recompute',architecture='native')
    with pytest.raises(ValueError):ConnectionConfig(variant='lookback_recompute',recompute_chunk_size=0)
    with pytest.raises(ValueError):ConnectionConfig(variant='lookback_recompute',recompute_input='embedding')
    retired=asdict(ConnectionConfig(variant='lookback_recompute'));retired.pop('recompute_input')
    with pytest.raises(ValueError,match='retired architecture'):ConnectionConfig.from_checkpoint(retired)
    retired=asdict(ConnectionConfig(variant='lookback_recompute'));retired.pop('recompute_memory')
    with pytest.raises(ValueError,match='retired architecture'):ConnectionConfig.from_checkpoint(retired)
    with pytest.raises(ValueError):ConnectionConfig(recompute_memory='per_layer')
    retired['recompute_memory']='shared_final_hidden'
    with pytest.raises(ValueError,match='retired architecture'):ConnectionConfig.from_checkpoint(retired)
    old_core=asdict(ConnectionConfig());old_core['recompute_memory']='shared_final_hidden'
    assert ConnectionConfig.from_checkpoint(old_core).variant=='baseline'
