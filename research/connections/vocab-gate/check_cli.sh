#!/bin/bash
set -euo pipefail
args=(--architecture vanilla --variant baseline --streams 1 --attention-backend fa2
      --vocab-gate --vocab-gate-alpha .5 --vocab-gate-chunk 2048
      --steps 3 --schedule-steps 305176 --microbatch 16 --seed 42
      --data-dir ../data --output-dir results/cli-check --checkpoint-dir ../diagnostic-checkpoints
      --run-id vocab-gate-debug --resume auto --segment-seconds 1800)
set +e
python -m scripts.connection_train "${args[@]}" --segment-steps 2
rc=$?
set -e
test "$rc" -eq 85
python -m scripts.connection_train "${args[@]}"
python - <<'PY'
import json
from pathlib import Path
p=Path('results/cli-check')
s=json.loads((p/'summary.json').read_text());c=json.loads((p/'config.json').read_text())
assert s['complete'] and s['steps']==3 and s['training_tokens']==3*32768
assert s['run_id']=='vocab-gate-debug' and c['logging_backend']=='local'
assert c['model']['vocab_gate'] and c['model']['vocab_size']==32768
rows=[json.loads(x) for x in (p/'steps.jsonl').read_text().splitlines()]
assert [r['step'] for r in rows]==[1,2,3]
Path('results/cli-check.json').write_text(json.dumps(dict(passed=True,resume_step=2,final_step=3,run_id=s['run_id'],no_wandb=True),indent=2)+'\n')
PY
rm -rf ../diagnostic-checkpoints
