"""Generate ephemeral tensors using an immutable pre-cleanup source checkout."""
import json
import os
from pathlib import Path
import sys
from dataclasses import asdict

from parity_helpers import METHOD, compare, evaluate

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import torch
from nanochat.connection_model import ConnectionConfig, ConnectionModel
from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.packed_training import seed_all

variant = sys.argv[2]
configure_attention('sdpa_cudnn')
torch.backends.cuda.matmul.allow_tf32 = False
seed_all(42)
c = ConnectionConfig(variant=variant, connection_backend='nvidia' if variant == 'mhc' else 'deepseek')
m = ConnectionModel(c).prepare(compile_blocks=False)
match_vanilla_initialization(m, 42)
with torch.no_grad():
    for b in m.blocks:
        b.attn.c_proj.weight.normal_(std=.01)
        b.mlp.c_proj.weight.normal_(std=.01)
x = torch.randperm(c.vocab_size, device='cuda')[:2048].reshape(1, -1)
y = (x + 1) % c.vocab_size
samples, repeat = {}, {}
for objective in ('dense_fp32', 'production_cce'):
    samples[objective] = evaluate(m, x, y, objective)
    again = evaluate(m, x, y, objective)
    repeat[objective] = compare(again, samples[objective], objective)
    del again
noise_path = Path('results/reference-noise')
noise_path.mkdir(parents=True, exist_ok=True)
(noise_path / (variant + '.json')).write_text(json.dumps(dict(source='699dcf2', method=METHOD,
    input_unique_tokens=x.unique().numel(), input_tokens=x.numel(), comparisons=repeat), indent=2) + '\n')
value = dict(config=asdict(c), model={k: v.detach().cpu() for k, v in m.state_dict().items()},
    x=x.cpu(), y=y.cpu(), samples=samples, repeat_control=repeat, method=METHOD,
    tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'), input_tokens_unique=x.unique().numel())
Path('../golden').mkdir(exist_ok=True)
torch.save(value, Path('../golden') / (variant + '.pt'))
print('Pinned 699dcf2 reference saved for ' + variant, flush=True)
