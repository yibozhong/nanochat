#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
run_id="${1:?missing run ID}"
recipe="${2:-chord}"
mkdir -p checkpoints results
if [[ -f results.tar || -f checkpoints.tar ]]; then
    [[ -f results.tar && -f checkpoints.tar ]]
    tar -xf results.tar
    tar -xf checkpoints.tar
fi
finish() {
    status=$?
    trap - EXIT
    if [[ $status != 0 && $status != 85 ]]; then
        echo "Native $recipe failed ($status), retaining GPU for debugging for 900 seconds" >&2
        sleep 900
    fi
    for dir in checkpoints results; do
        tar -cf "$TASK_ROOT/$dir.tar.tmp" -C "$TASK_ROOT" "$dir" && mv "$TASK_ROOT/$dir.tar.tmp" "$TASK_ROOT/$dir.tar" || exit 74
    done
    exit "$status"
}
trap finish EXIT
/opt/project/.venv/bin/python - "$run_id" "$recipe" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path('source.json').read_text())
assert s['run_id'] == sys.argv[1] and s['steps'] == 305176 and s['seed'] == 42
assert s['architecture'] == 'native' and s['initialization'] == 'fresh'
if sys.argv[2] == 'mhc':
    assert s['connection_variant'] == 'mhc' and s['streams'] == 4
elif sys.argv[2] in ('token_vnorm', 'token_bounded'):
    assert s['native_chord_write'] == sys.argv[2]
    if sys.argv[2] == 'token_bounded':
        assert s['gate_input_norm'] == 'joint_h_v' and s['gate_joint_rms_eps'] == 1e-6
    else:
        assert s['gate_velocity_rms_eps'] == 1e-6
    assert '--native-chord-write=' + sys.argv[2] in s['model_args']
else:
    assert sys.argv[2] == 'chord' and s['native_chord_write'] == 'three_scalar'
assert isinstance(s['model_args'], list) and all(a.startswith('--') for a in s['model_args'])
Path('model.args').write_text('\n'.join(s['model_args']) + '\n')
if Path('results/train/summary.json').exists():
    r = json.loads(Path('results/train/summary.json').read_text())
    assert r['complete'] and r['source'] == s and r['steps'] == s['steps'] and r['run_id'] == s['run_id']
    for name, meta in r['checkpoint_files'].items():
        assert Path('checkpoints', name).stat().st_size == meta['bytes']
elif Path('results/train/progress.json').exists():
    assert all(Path('checkpoints', n).stat().st_size for n in ('latest.pt', 'best.pt'))
PY
[[ ! -f results/train/summary.json ]] || exit 0
tar -xf source.tar
source nanochat-src/research/connections/runtime/packed_worker_env.sh
export HF_HOME="$TASK_ROOT/fa3-cache" NANOCHAT_FA3_REVISION=de87b9b5af06dd9984df595bef90b2eba44b181a
unset NANOCHAT_DISABLE_FA3
HF_HUB_OFFLINE=0 python - <<'PY'
import os
from kernels import get_kernel
assert hasattr(get_kernel('varunneal/flash-attention-3', revision=os.environ['NANOCHAT_FA3_REVISION']).flash_attn_interface, 'flash_attn_func')
PY
mapfile -t model_args < "$TASK_ROOT/model.args"
training_args=(--schedule-steps 305176 --seed 42 --microbatch 16 --data-dir ../data)
if [[ "$recipe" == mhc ]]; then
    group=native-mhc-normal-init
    WANDB_MODE=disabled timeout --kill-after=30s 2400 python -m research.connections.ngpt.connection_init_preflight "${model_args[@]}" "${training_args[@]}"
elif [[ "$recipe" == token_vnorm || "$recipe" == token_bounded ]]; then
    mode="${recipe//_/-}"
    group="native-chord-token-controls-${recipe#token_}"
    if [[ ! -f "results/native-chord-$mode-sm90.json" ]]; then
        WANDB_MODE=disabled timeout --kill-after=30s 2400 python -m research.connections.native.chord.preflight "--$mode"
    fi
else
    group=native-chord-three-scalar
    if [[ ! -f results/native-chord-sm90.json ]]; then
        WANDB_MODE=disabled timeout --kill-after=30s 2400 python -m research.connections.native.chord.preflight
    fi
fi
echo "FORMAL_TRAINING_BEGIN $group $run_id"
python -m scripts.connection_train "${model_args[@]}" "${training_args[@]}" --steps 305176 \
    --evaluate-test --output-dir results/train --checkpoint-dir ../checkpoints \
    --formal-training --wandb-group "$group" --run-id "$run_id" --resume auto --segment-seconds 1800
