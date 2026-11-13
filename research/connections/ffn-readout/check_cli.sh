#!/bin/bash
set -euo pipefail
# Run only in the allocated GPU sandbox, from nanochat-src. No W&B.
args=(--variant baseline --ffn-readout-width 64 --ffn-readout-init gradmax
      --ffn-readout-backend triton --steps 3 --schedule-steps 305176
      --attention-backend fa2 --microbatch 16 --streams 1 --seed 42
      --data-dir ../data --output-dir ../results/readout/cli
      --checkpoint-dir ../diagnostic-checkpoints --resume auto
      --segment-steps 2 --run-id local-ffn-readout-resume)
test ! -e ../diagnostic-checkpoints/latest.pt
set +e
python -m scripts.connection_train "${args[@]}" > ../results/readout/cli-first.log 2>&1
status=$?
set -e
test "$status" -eq 85
python - <<'PY'
import json
from pathlib import Path
p=Path('../results/readout/cli')
assert json.loads((p/'progress.json').read_text())['step']==2
assert json.loads((p/'config.json').read_text())['logging_backend']=='local'
(p/'init-before-resume.json').write_text((p/'readout-initialization.json').read_text())
PY
python -m scripts.connection_train "${args[@]}" > ../results/readout/cli-resume.log 2>&1
python - <<'PY'
import json
from pathlib import Path
import torch
p=Path('../results/readout/cli')
s=json.loads((p/'summary.json').read_text())
assert s['complete'] and s['steps']==3 and s['training_tokens']==3*32768
assert s['run_id']=='local-ffn-readout-resume'
assert json.loads((p/'config.json').read_text())['readout_initialization'] is None
assert (p/'init-before-resume.json').read_text()==(p/'readout-initialization.json').read_text()
state=torch.load('../diagnostic-checkpoints/latest.pt',map_location='cpu',weights_only=False)
assert state['step']==3 and state['config']['ffn_readout_width']==64
assert state['optimizer'] and state['rng'] and state['run_id']==s['run_id']
assert torch.count_nonzero(state['model']['blocks.7.mlp.c_proj.weight'][512:])>0
report=dict(passed=True,first_segment_exit=85,resumed_step=3,training_tokens=3*32768,
            initialization_skipped_on_resume=True,formal_training=False,wandb_created=False,
            model_optimizer_rng_present=True,data_position='step 3 * 16 packed rows')
Path('../results/readout/cli-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
PY
