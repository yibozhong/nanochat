"""Exact saved optimizer/RNG and matched common weights on the new backend."""
from dataclasses import asdict
import pytest
import torch
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for,schedule
from nanochat.packed_training import seed_all,snapshot,restore_rng
from test_connection_training_resume import equal_tree

def build():
    seed_all(42)
    return ConnectionModel(ConnectionConfig(attention_kernel='fla_triton',
        depth=2,width=128,heads=1,vocab_size=256,sequence_length=128)).prepare()

def test_exact_resume_and_fixed_gradient_update(tmp_path):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.recompile_limit=64
    model=build();opt,wd=optimizer_for(model)
    x=torch.randint(0,256,(2,128),device='cuda');y=(x+1)%256
    for step in range(3):
        opt.zero_grad(set_to_none=True);schedule(opt,step,305176,wd)
        model(x,y).backward();opt.step()
    gradients={n:p.grad.clone() for n,p in model.named_parameters()}
    path=tmp_path/'checkpoint.pt';torch.save(snapshot(model,opt,3,config=asdict(model.config)),path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    other=build();other_opt,_=optimizer_for(other)
    other.load_state_dict(state['model']);other_opt.load_state_dict(state['optimizer'])
    equal_tree(model.state_dict(),other.state_dict());equal_tree(opt.state_dict(),other_opt.state_dict())
    restore_rng(state['rng']);expected=(torch.rand(8),torch.rand(8,device='cuda'))
    restore_rng(state['rng']);equal_tree(expected,(torch.rand(8),torch.rand(8,device='cuda')))
    for m,o in ((model,opt),(other,other_opt)):
        for n,p in m.named_parameters():p.grad=gradients[n].clone()
        schedule(o,3,305176,wd);o.step()
    equal_tree(model.state_dict(),other.state_dict());equal_tree(opt.state_dict(),other_opt.state_dict())
