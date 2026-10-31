"""Generate ephemeral tensors using an immutable pre-cleanup source checkout."""
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(sys.argv[1]).resolve()))
import torch
from dataclasses import asdict
from nanochat.connection_model import ConnectionConfig,ConnectionModel
from nanochat.connection_experiment import configure_attention,match_vanilla_initialization
from nanochat.packed_training import seed_all

variant=sys.argv[2]
configure_attention('sdpa_cudnn');torch.backends.cuda.matmul.allow_tf32=False;seed_all(42)
c=ConnectionConfig(variant=variant,connection_backend='nvidia' if variant=='mhc' else 'deepseek')
m=ConnectionModel(c).prepare(compile_blocks=False);match_vanilla_initialization(m,42)
# Preserve the realistic initialization while exercise nonzero residual branches.
with torch.no_grad():
    for b in m.blocks:
        b.attn.c_proj.weight.normal_(std=.01);b.mlp.c_proj.weight.normal_(std=.01)
x=torch.randint(0,c.vocab_size,(1,2048),device='cuda');y=(x+1)%c.vocab_size
loss=m(x,y);loss.backward()
value=dict(config=asdict(c),model={k:v.detach().cpu() for k,v in m.state_dict().items()},
    x=x.cpu(),y=y.cpu(),loss=loss.detach().cpu(),
    grads={k:p.grad.cpu() for k,p in m.named_parameters()},
    tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'))
Path('../golden').mkdir(exist_ok=True)
torch.save(value,Path('../golden')/(variant+'.pt'))
print('Pinned 699dcf2 reference saved for '+variant,flush=True)
