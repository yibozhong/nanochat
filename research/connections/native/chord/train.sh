#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
run_id="${1:?missing run ID}"
recipe="${2:?missing recipe}"
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
        echo "$recipe failed ($status), retaining GPU for debugging for 900 seconds" >&2
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
recipe = sys.argv[2]
assert recipe == s['recipe'] and recipe in ('vanilla_baseline', 'vanilla_baseline_zero', 'mhc', 'chord_scalar_bf16', 'chord_next_bf16')
assert s['run_id'] == sys.argv[1] and s['schedule_steps'] == 30518 and s['seed'] == 42
assert s['steps'] in (0, s['schedule_steps']) and s['wandb_project'] == 'nanochat-1b'
assert s['initialization'] == 'fresh' and s['formal_training'] == bool(s['steps'])
if recipe == 'mhc':
    assert (s['architecture'], s['attention_backend'], s['connection_variant'], s['streams']) == ('native', 'fa3', 'mhc', 4)
else:
    assert (s['architecture'], s['attention_backend']) == ('vanilla', 'fa2')
    flags = set(s['model_args'])
    assert '--residual-projection-init=' + s['residual_projection_init'] in flags
    if recipe in ('vanilla_baseline', 'vanilla_baseline_zero'):
        assert s['model_variant'] == 'vanilla' and '--variant=baseline' in flags
    else:
        assert {'--chord-state-dtype=bfloat16', '--chord-velocity-geometry=project_only',
                '--ngpt-velocity-mode=lookahead_split_learned_beta', '--velocity-init=' + s['velocity_init']} <= flags
        if recipe == 'chord_next_bf16':
            assert (s['velocity_init'], s['residual_projection_init']) == ('projected_h0', 'zero')
            assert {'--chord-update-based', '--chord-soft-force-tau=0.05', '--chord-six-controls=learned_dynamic'} <= flags
        else:
            assert (s['velocity_init'], s['residual_projection_init']) == ('token', 'normal')
            assert '--chord-update-based' not in flags and not any(o.startswith('--chord-six-controls=') for o in flags)
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
if [[ "$recipe" != mhc ]]; then
    export NANOCHAT_DISABLE_FA3=1
else
    export HF_HOME="$TASK_ROOT/fa3-cache" NANOCHAT_FA3_REVISION=de87b9b5af06dd9984df595bef90b2eba44b181a
    unset NANOCHAT_DISABLE_FA3
    HF_HUB_OFFLINE=0 python - <<'PY'
import os
from kernels import get_kernel
assert hasattr(get_kernel('varunneal/flash-attention-3', revision=os.environ['NANOCHAT_FA3_REVISION']).flash_attn_interface, 'flash_attn_func')
PY
fi
mapfile -t model_args < "$TASK_ROOT/model.args"
steps=$(python -c "import json; print(json.load(open('source.json'))['steps'])")
schedule=$(python -c "import json; print(json.load(open('source.json'))['schedule_steps'])")
group=$(python -c "import json; print(json.load(open('source.json'))['group'])")
training_args=(--schedule-steps "$schedule" --seed 42 --microbatch 16 --data-dir ../data)
if [[ "$recipe" == chord_next_bf16 ]]; then
    if [[ ! -f results/vanilla-next-soft-projected-h0-sm90.json ]]; then
        WANDB_MODE=disabled timeout --kill-after=30s 2400 python -m research.connections.ngpt.next_soft_preflight
    fi
elif [[ "$recipe" == mhc ]]; then
    WANDB_MODE=disabled timeout --kill-after=30s 2400 python -m research.connections.ngpt.connection_init_preflight "${model_args[@]}" "${training_args[@]}"
else
    if [[ ! -f results/baseline-check/summary.json ]]; then
        WANDB_MODE=disabled python -m scripts.connection_train "${model_args[@]}" "${training_args[@]}" --steps 2 \
            --output-dir results/baseline-check --checkpoint-dir ../preflight-checkpoints
    fi
fi
[[ "$steps" != 0 ]] || exit 0
echo "FORMAL_TRAINING_BEGIN $group $run_id"
python -m scripts.connection_train "${model_args[@]}" "${training_args[@]}" --steps "$steps" \
    --evaluate-test --output-dir results/train --checkpoint-dir ../checkpoints \
    --formal-training --wandb-group "$group" --run-id "$run_id" --resume auto --segment-seconds 1800
