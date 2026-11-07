#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
TASK_MODE="${1:?gate or train}"
TASK_RUN_ID="${2:-hash-vo-gate}"
TASK_VARIANT=hash_vo
TASK_STEPS=305176
mkdir -p checkpoints results work cache
if [ -f results.tar ]; then tar -xf results.tar; fi
if [ -f checkpoints.tar ]; then tar -xf checkpoints.tar; fi
checkpoint_archives_on_exit() {
    TASK_EXIT_STATUS=$?
    trap - EXIT
    if ! tar -cf "$TASK_ROOT/checkpoints.tar.tmp" -C "$TASK_ROOT" checkpoints ||
       ! mv "$TASK_ROOT/checkpoints.tar.tmp" "$TASK_ROOT/checkpoints.tar" ||
       ! tar -cf "$TASK_ROOT/results.tar.tmp" -C "$TASK_ROOT" results ||
       ! mv "$TASK_ROOT/results.tar.tmp" "$TASK_ROOT/results.tar"; then
        exit 74
    fi
    exit "$TASK_EXIT_STATUS"
}
trap checkpoint_archives_on_exit EXIT
cp "$0" results/launcher.sh
sha256sum -c inputs.sha256
if [ "$TASK_MODE" = train ] && [ -f results/train/summary.json ]; then
    /opt/project/.venv/bin/python - "$TASK_RUN_ID" <<'PYDONE'
import json,sys
from pathlib import Path
r=json.loads(Path('results/train/summary.json').read_text())
assert r['complete'] and r['source']==json.loads(Path('source.json').read_text())
assert r['variant']=='hash_vo' and r['run_id']==sys.argv[1] and r['steps']==305176
for n,m in r['checkpoint_files'].items():assert Path('checkpoints',n).stat().st_size==m['bytes']
PYDONE
    exit 0
fi
tar -xf source.tar
mkdir -p work/preflight-data
tar -xf gate-data.tar -C work/preflight-data
if [ "$TASK_MODE" = train ]; then
    tar -xf nanochat-2k-10b-data-7685535.tar
    export WANDB_MODE=online
elif [ "$TASK_MODE" = gate ]; then
    export WANDB_MODE=disabled
else
    echo "Invalid mode" >&2; exit 2
fi
export HASH_WORKER_INVOCATION
HASH_WORKER_INVOCATION=$(/opt/project/.venv/bin/python -c 'import uuid; print(uuid.uuid4().hex)')
source nanochat-src/research/connections/hash-vo/runtime.sh
python - <<'PYBOOT'
import json,os,time
from pathlib import Path
with Path('results/worker-starts.jsonl').open('a') as f:
    f.write(json.dumps(dict(invocation=os.environ['HASH_WORKER_INVOCATION'],scratch=str(Path.cwd()),time=time.time()))+'\n')
PYBOOT
cd nanochat-src
ln -sfnT ../results results
cp ../source.json source.json
TASK_GPU_FAMILY=$(python - <<'HOPPER'
import json,torch
from pathlib import Path
s=json.loads(Path('source.json').read_text())
assert s['model_variant']=='hash_vo' and s['initialization']=='seed42_from_zero_no_validation_weights'
assert torch.cuda.get_device_capability()==(9,0)
name=torch.cuda.get_device_name()
assert 'H100' in name or 'H200' in name
print('H100' if 'H100' in name else 'H200')
HOPPER
)
if [ "$TASK_MODE" = gate ]; then
    python -m scripts.hash_vo_gate
else
    WANDB_MODE=disabled python -m scripts.hash_vo_preflight --family "$TASK_GPU_FAMILY"
    echo "FORMAL_TRAINING_BEGIN $TASK_VARIANT $TASK_RUN_ID"
    python -m scripts.hash_vo_train --attention-backend hardware_auto \
        --steps "$TASK_STEPS" --schedule-steps "$TASK_STEPS" --microbatch 16 --seed 42 \
        --no-activation-checkpoint --evaluate-test \
        --data-dir ../data --output-dir results/train --checkpoint-dir ../checkpoints \
        --formal-training --wandb-group hash-vo-10b-20261107 --run-id "$TASK_RUN_ID" \
        --resume auto --segment-seconds 1800
fi
