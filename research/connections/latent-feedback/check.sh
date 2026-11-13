#!/bin/bash
set -euo pipefail
python -m pytest -q tests/test_latent_feedback.py --junitxml=results/tests.xml
python -m scripts.connection_feedback_debug --data-dir ../data --output-dir results/profile
# Actual CLI transport: exit85 checkpoint -> same protocol/run/data position.
TASK_ARGS=(--variant baseline --streams 1 --latent-feedback --attention-backend fa2
    --steps 3 --schedule-steps 305176 --microbatch 16 --seed 42 --data-dir ../data
    --run-id feedback-local-debug --segment-seconds 1800)
set +e
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../resume-check --segment-steps 2
TASK_STATUS=$?
set -e
test "$TASK_STATUS" -eq 85
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../resume-check --resume auto
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../continuous-check
python - <<'PY'
import json,sys,torch
from pathlib import Path
sys.path.insert(0,'tests')
from test_connection_training_resume import equal_tree
a=torch.load('../resume-check/checkpoints/latest.pt',map_location='cpu',weights_only=False)
b=torch.load('../continuous-check/checkpoints/latest.pt',map_location='cpu',weights_only=False)
assert a['step']==b['step']==3
for key in ('model','optimizer','rng','config','protocol','run_id','data_manifest'):
    equal_tree(a[key],b[key])
rows=[json.loads(line) for line in Path('../resume-check/steps.jsonl').read_text().splitlines()]
assert [r['step'] for r in rows]==[1,2,3]
Path('results/cli-resume.json').write_text(json.dumps(dict(complete=True,step=3,exit85=True,
    checks=['model','optimizer','rng','config','protocol','run_id','data_position'],
    passes=[r['passes'] for r in rows],formal_training=False),indent=2)+'\n')
PY
rm -r ../resume-check ../continuous-check
echo 'FEEDBACK_DEBUG_COMPLETE'
