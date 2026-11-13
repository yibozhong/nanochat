"""Compare task and rate pressure on a trained confidence router, without updates."""
import json
import os
from pathlib import Path
import torch
from nanochat.connection_model import ConnectionConfig, build_connection_model
from nanochat.connection_experiment import configure_attention
from nanochat.packed_training import PackedData

assert os.environ['WANDB_MODE'] == 'disabled'
torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
configure_attention('fa2')
state = torch.load('../diagnostic-state/latest.pt', map_location='cpu', weights_only=False)
assert state['formal_training'] is False
c = ConnectionConfig(**state['config'])
assert c.hca_dynamic_gradient == 'confidence' and not c.hca_noise
model = build_connection_model(c).prepare(compile_blocks=False).train()
model.load_state_dict(state['model'])
report = dict(step=state['step'], formal_training=False, config=state['config'],
              checkpoint_source=state['source'], source=json.loads(Path('source.json').read_text()))
step = state['step']; del state
data = PackedData('../data')
routers = [(n,p) for n,p in model.named_parameters() if '.boundary.' in n]
rows = []
for split, start in [('train', step*16), ('dev', 0)]:
    x,y = data.batch(split, start, 16)
    values = model(x,y,return_loss_components=True)
    ce = torch.autograd.grad(values['ce'], [p for _,p in routers], retain_graph=True)
    rate = torch.autograd.grad(values['compression_loss'], [p for _,p in routers])
    row = dict(split=split, data_row=start, values={k:v.item() for k,v in values.items()}, parameters=[])
    for (name,_), cg, rg in zip(routers, ce, rate):
        row['parameters'].append(dict(name=name, ce_norm=cg.norm().item(),
            rate_norm=rg.norm().item(), weighted_rate_norm=(c.hca_ratio_loss_weight*rg).norm().item(),
            cosine=torch.nn.functional.cosine_similarity(cg.flatten(),rg.flatten(),dim=0).item(),
            ce_sum=cg.sum().item(), rate_sum=rg.sum().item()))
    rows.append(row)
report['samples'] = rows
Path('results/optimization/confidence-rate-gradients.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report),flush=True)
