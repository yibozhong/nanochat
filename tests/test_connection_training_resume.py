"""Matched initialization and optimizer/RNG checkpoint invariants for pretraining."""
from dataclasses import asdict

import numpy as np
import pytest
import torch

from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.packed_training import seed_all, snapshot, restore_rng


def equal_tree(a,b):
    assert type(a) is type(b),(type(a),type(b))
    if isinstance(a,torch.Tensor):
        assert a.dtype==b.dtype and a.shape==b.shape
        assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):
        assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal_tree(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    else:
        assert a==b


def build(variant):
    seed_all(42)
    config=ConnectionConfig(variant=variant,depth=2,width=128,heads=1,
        sequence_length=128,vocab_size=256,
        connection_backend='nvidia')
    model=ConnectionModel(config).prepare()
    match_vanilla_initialization(model,42)
    return model


def test_backbones_have_identical_initial_weights():
    configure_attention('sdpa_cudnn')
    base=build('baseline');mhc=build('mhc')
    left,right=base.state_dict(),mhc.state_dict()
    assert all(torch.equal(value,right[name]) for name,value in left.items())


@pytest.mark.parametrize('variant',['baseline','mhc'])
def test_optimizer_rng_checkpoint_and_fixed_gradient_next_update(variant,tmp_path):
    configure_attention('sdpa_cudnn');torch.backends.cuda.matmul.allow_tf32=False
    model=build(variant);opt,wd=optimizer_for(model)
    x=torch.randint(0,256,(2,128),device='cuda');y=(x+1)%256
    for step in range(4):
        schedule(opt,step,30518,wd);opt.zero_grad(set_to_none=True)
        model(x,y).backward();opt.step()
    gradients={name:p.grad.detach().clone() for name,p in model.named_parameters()}
    state=snapshot(model,opt,4,config=asdict(model.config),next_data_row=4*16)
    path=tmp_path/'checkpoint.pt';torch.save(state,path)
    loaded=torch.load(path,map_location='cpu',weights_only=False)
    other=build(variant);other_opt,other_wd=optimizer_for(other)
    other.load_state_dict(loaded['model']);other_opt.load_state_dict(loaded['optimizer'])
    equal_tree(model.state_dict(),other.state_dict())
    equal_tree(opt.state_dict(),other_opt.state_dict())
    assert loaded['step']==4 and loaded['next_data_row']==64 and wd==other_wd
    restore_rng(state['rng'])
    expected=(torch.rand(8),torch.rand(8,device='cuda'),np.random.rand(8))
    restore_rng(loaded['rng'])
    actual=(torch.rand(8),torch.rand(8,device='cuda'),np.random.rand(8))
    equal_tree(expected,actual)
    for target,optimizer in ((model,opt),(other,other_opt)):
        for name,p in target.named_parameters():p.grad=gradients[name].clone()
        schedule(optimizer,4,30518,wd);optimizer.step()
    equal_tree(model.state_dict(),other.state_dict())
    equal_tree(opt.state_dict(),other_opt.state_dict())
