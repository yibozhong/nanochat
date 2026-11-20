#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
TASK_RUN_ID="$1"
TASK_MODE="$2"
TASK_VELOCITY_MODE="${3:-ambient}"
TASK_MOMENTUM_ARGS=()
TASK_GROUP=residual-nonzero-10b-20261117
case "$TASK_MODE/$TASK_VELOCITY_MODE" in
    ngpt_hbformer/transport_rms|ngpt_hbformer/transport_layernorm)
        TASK_MOMENTUM_ARGS=(--velocity-init token_position --velocity-embedding-std .02
            --momentum-beta-init .5 --momentum-gamma-init 0.6931471805599453 --momentum-eta-init 1
            --momentum-velocity-rms-floor .1)
        TASK_GROUP=ngpt-hbformer-10b
        ;;
    heavy_ball_rms/ambient|ngpt_heavy_ball/transport_rms)
        TASK_MOMENTUM_ARGS=(--velocity-init token --velocity-embedding-std .02
            --momentum-beta-init .5 --momentum-gamma-init .5 --momentum-eta-init .05
            --momentum-velocity-rms-floor .1)
        if [ "$TASK_MODE" = heavy_ball_rms ]; then
            TASK_GROUP=heavy-ball-rms-10b
        else
            TASK_GROUP=ngpt-momentum-transport-rms-10b
        fi
        ;;
    lookahead_only/ambient)
        TASK_MOMENTUM_ARGS=(--velocity-init token_position --velocity-embedding-std .02 --momentum-beta-init .5 --lookahead-mu-init 0)
        TASK_GROUP=lookahead-only-10b
        ;;
    ngpt_lookahead/lookahead_transport|ngpt_lookahead/lookahead_axis_write|ngpt_lookahead/lookahead_split_write|ngpt_lookahead/lookahead_split_learned_beta|ngpt_lookahead/lookahead_geodesic_split)
        TASK_MOMENTUM_ARGS=(--velocity-init token_position --velocity-embedding-std .02 --momentum-beta-init .5 --lookahead-mu-init 0)
        TASK_GROUP=ngpt-lookahead-5k
        if [[ "$TASK_VELOCITY_MODE" == lookahead_axis_write ]]; then TASK_GROUP=arc-residual-axis-write-10b; fi
        if [[ "$TASK_VELOCITY_MODE" == lookahead_split_write ]]; then TASK_GROUP=arc-residual-split-write-10b; fi
        if [[ "$TASK_VELOCITY_MODE" == lookahead_split_learned_beta ]]; then TASK_GROUP=arc-residual-split-learned-beta-10b; fi
        if [[ "$TASK_VELOCITY_MODE" == lookahead_geodesic_split ]]; then TASK_GROUP=arc-residual-geodesic-split-10b; fi
        ;;
    vanilla/ambient|ngpt/ambient|ngpt_heavy_ball/ambient) ;;
    *) exit 64 ;;
esac
TASK_SCHEDULE_STEPS=305176
TASK_STEPS=$TASK_SCHEDULE_STEPS
if [[ "$TASK_VELOCITY_MODE" == lookahead_transport ]]; then TASK_STEPS=5000; fi
if [[ "${4:-}" == --extend-to-schedule || "${4:-}" == --fresh-10b ]]; then
    [[ "$#" == 4 && "$TASK_MODE" == ngpt_lookahead ]]
    TASK_GROUP=ngpt-lookahead-10b
    TASK_STEPS=$TASK_SCHEDULE_STEPS
elif [[ "${4:-}" == --arc || "${4:-}" == --arc-muon || "${4:-}" == --arc-muon-radius ]]; then
    [[ "$TASK_MODE/$TASK_VELOCITY_MODE" == vanilla/ambient ]]
    TASK_ARC_MU=1
    if [[ "$#" == 5 && "$4" == --arc-muon-radius && "$5" == --no-arc-lookahead ]]; then
        TASK_ARC_MU=0
    else
        [[ "$#" == 4 ]]
    fi
    TASK_MOMENTUM_ARGS=(--matrix-optimizer arc --arc-mu "$TASK_ARC_MU" --arc-beta .5)
    TASK_GROUP=arc-optimizer-10b
    if [[ "$4" == --arc-muon* ]]; then
        TASK_MOMENTUM_ARGS+=(--arc-direction muon)
        TASK_GROUP=arc-muon-direction-optimizer-10b
    fi
    if [[ "$4" == --arc-muon-radius ]]; then
        TASK_MOMENTUM_ARGS+=(--arc-step-scale radius)
        TASK_GROUP=arc-muon-direction-radius-step-10b
        if [[ "$TASK_ARC_MU" == 0 ]]; then TASK_GROUP=arc-muon-direction-radius-step-mu0-10b; fi
    fi
elif [[ "${4:-}" == --token-chord-gates ]]; then
    [[ "$#" == 4 && "$TASK_VELOCITY_MODE" == lookahead_split_learned_beta ]]
    TASK_MOMENTUM_ARGS+=(--token-chord-gates)
    TASK_GROUP=chord-token-alpha-gates-10b
elif [[ "${4:-}" == --token-beta-gate ]]; then
    [[ "$#" == 4 && "$TASK_VELOCITY_MODE" == lookahead_split_learned_beta ]]
    TASK_MOMENTUM_ARGS+=(--token-beta-gate)
    TASK_GROUP=chord-token-beta-gate-10b
elif [[ "${4:-}" == --signed-chord-steps ]]; then
    [[ "$#" == 4 && "$TASK_VELOCITY_MODE" == lookahead_split_learned_beta ]]
    TASK_MOMENTUM_ARGS+=(--signed-chord-steps)
    TASK_GROUP=chord-signed-steps-learned-beta-10b
elif [ "$#" -gt 3 ]; then
    [[ "$#" == 4 && "$TASK_MODE" == ngpt_hbformer && "$4" == --no-ngpt-weight-norm ]]
    TASK_MOMENTUM_ARGS+=("$4")
    if [[ "$TASK_VELOCITY_MODE" == transport_layernorm ]]; then
        TASK_GROUP=ngpt-hbformer-velocity-layernorm-10b
    else
        TASK_STEPS=5000
        TASK_GROUP=ngpt-hbformer-no-weight-norm-5k
    fi
fi
mkdir -p checkpoints results
if [ -f results.tar ]; then tar -xf results.tar; fi
if [ -f checkpoints.tar ]; then tar -xf checkpoints.tar; fi
finish() {
    TASK_STATUS=$?
    trap - EXIT
    if [ "$TASK_STATUS" -ne 0 ] && [ "$TASK_STATUS" -ne 85 ]; then
        echo "TRAINING_FAILED status=$TASK_STATUS; keeping debug allocation for 900 seconds" >&2
        sleep 900
    fi
    tar -cf "$TASK_ROOT/checkpoints.tar.tmp" -C "$TASK_ROOT" checkpoints || exit 74
    mv "$TASK_ROOT/checkpoints.tar.tmp" "$TASK_ROOT/checkpoints.tar" || exit 74
    tar -cf "$TASK_ROOT/results.tar.tmp" -C "$TASK_ROOT" results || exit 74
    mv "$TASK_ROOT/results.tar.tmp" "$TASK_ROOT/results.tar" || exit 74
    exit "$TASK_STATUS"
}
trap finish EXIT
if [[ "${4:-}" == --extend-to-schedule ]]; then
    if [[ -f checkpoints.tar || -f results.tar ]]; then
        [[ -f checkpoints.tar && -f results.tar ]]
    else
        tar -xf parent-checkpoints.tar checkpoints/latest.pt checkpoints/best.pt
        tar -xf parent-results.tar results/train/progress.json results/train/steps.jsonl
    fi
    tar -xOf parent-results.tar results/ngpt-sm90.json > parent-validation.json
fi
cp "$0" results/launcher.sh
/opt/project/.venv/bin/python - "$TASK_RUN_ID" "$TASK_MODE" "$TASK_VELOCITY_MODE" "$TASK_STEPS" "$TASK_SCHEDULE_STEPS" "${4:-}" "${TASK_ARC_MU:-}" <<'PY'
import json,sys
from pathlib import Path
s=json.loads(Path('source.json').read_text())
assert s['formal_training'] and s['run_id']==sys.argv[1]
assert s['steps']==int(sys.argv[4]) and s['training_tokens']==s['steps']*32768
assert s.get('schedule_steps',305176)==int(sys.argv[5])==305176
assert s['initialization']=='seed42_from_scratch_no_diagnostic_weights'
assert s['attention_backend']=='fa2' and s['residual_dynamics']==sys.argv[2]
assert s['model']==dict(depth=8,width=512,heads=4,sequence_length=2048,vocab_size=32768)
assert s['velocity_embedding']==(sys.argv[2] in ('ngpt_heavy_ball','heavy_ball_rms','ngpt_hbformer','ngpt_lookahead','lookahead_only')) and s['ngpt_alpha_init']==.05
assert s['residual_projection_init']=='normal' and s['residual_projection_init_std']==.02
assert s['ngpt_force_rms_floor']==1e-6
assert s.get('ngpt_velocity_mode','ambient')==sys.argv[3]
signed_steps = sys.argv[6]=='--signed-chord-steps'
assert signed_steps == s.get('signed_chord_steps',False)
token_gates = sys.argv[6]=='--token-chord-gates'
assert token_gates == s.get('token_chord_gates',False)
token_beta_gate = sys.argv[6]=='--token-beta-gate'
assert token_beta_gate == s.get('token_beta_gate',False)
if token_gates or token_beta_gate:
    assert not signed_steps and s['paired_run']=='6721b437' and s['residual_kernel']=='fused_chord_triton'
assert (sys.argv[6] in ('--arc','--arc-muon','--arc-muon-radius')) == (s.get('matrix_optimizer')=='arc')
assert (sys.argv[6] in ('--arc-muon','--arc-muon-radius')) == (s.get('arc_direction')=='muon')
assert (sys.argv[6]=='--arc-muon-radius') == (s.get('arc_step_scale')=='radius')
if s.get('matrix_optimizer')=='arc':
    assert s['steps']==s['schedule_steps']==305176 and s['reference_run']=='2eac64db'
    assert s['arc_mu']==int(sys.argv[7]) and s['arc_beta']==.5 and not s['velocity_embedding']
    assert s['residual_dynamics']=='vanilla' and 'continuation' not in s
if sys.argv[2]=='lookahead_only':
    assert s['steps']==s['schedule_steps']==305176 and not s['ngpt_weight_norm'] and 'continuation' not in s
    assert s['residual_write']=='vanilla_addition' and s['ngpt_velocity_mode']=='ambient'
    assert s['lookahead_mu_init']==0 and s['momentum_beta_init']==.5 and not s['momentum_beta_trainable']
    assert s['velocity_init']=='token_position' and s['velocity_embedding_std']==.02
elif sys.argv[2]=='ngpt_lookahead':
    extended = sys.argv[6]=='--extend-to-schedule'
    assert extended == ('continuation' in s)
    learned_beta = sys.argv[3]=='lookahead_split_learned_beta'
    axis_write = learned_beta or sys.argv[3] in ('lookahead_axis_write','lookahead_split_write','lookahead_geodesic_split')
    assert s['steps']==(305176 if extended or sys.argv[6]=='--fresh-10b' or axis_write else 5000) and not s['ngpt_weight_norm']
    if axis_write:
        assert not sys.argv[6] or (signed_steps or token_gates or token_beta_gate) and learned_beta
        assert s['reference_run']==('6721b437' if token_gates or token_beta_gate else 'd4293868')
        if learned_beta or sys.argv[3] in ('lookahead_split_write','lookahead_geodesic_split'):
            assert s['parallel_step_init']==s['perpendicular_step_init']==s['ngpt_alpha_init']
            expected = 'independent_softplus_with_linear_lower_extension' if signed_steps else 'independent_softplus'
            assert s['step_parameterization']==expected and 'kappa_init' not in s
        else:
            assert s['kappa_init']==0
    assert s['lookahead_mu_init']==0 and s['momentum_beta_init']==.5 and s['momentum_beta_trainable']==learned_beta
    if learned_beta:
        assert s['learned_beta_sites']==2*s['model']['depth']-1 and s['beta_parameterization']=='sigmoid'
    assert s['velocity_init']=='token_position' and s['velocity_embedding_std']==.02
elif sys.argv[2]=='ngpt_hbformer':
    import math
    assert s['momentum_eta_init']==1 and not s['momentum_eta_trainable']
    assert s['ngpt_velocity_mode'] in ('transport_rms','transport_layernorm')
    assert s['velocity_init']=='token_position' and s['velocity_embedding_std']==.02
    assert s['momentum_beta_init']==.5 and s['momentum_gamma_init']==math.log(2.)
    normalized = sys.argv[6] != '--no-ngpt-weight-norm'
    assert s.get('ngpt_weight_norm',True) == normalized
    assert s['backbone_weight_constraint']==('ngpt_unit_l2_init_and_post_step' if normalized else 'none')
    if s['ngpt_velocity_mode']=='transport_layernorm':
        assert not normalized and s['steps']==305176
        assert not s['velocity_layernorm_bias'] and s['velocity_layernorm_eps']==1e-5 and s['velocity_gain_init']==1
    else:
        assert s['steps']==(305176 if normalized else 5000)
elif sys.argv[3]=='transport_rms' or sys.argv[2]=='heavy_ball_rms':
    assert s['momentum_eta_init']==.05 and s['momentum_velocity_rms_floor']==.1
p=Path('results/train/progress.json')
if p.exists():
    r=json.loads(p.read_text())
    assert r['run_id']==s['run_id']
    if r['source'] != s:
        parent=s['continuation']
        assert r['source']==parent['source'] and r['step']==parent['step'] and r['tokens']==32768*r['step']
        for name, meta in parent['checkpoint_files'].items():
            assert Path('checkpoints',name).stat().st_size==meta['bytes']
        r['source']=s
        p.write_text(json.dumps(r,indent=2)+'\n')
    assert Path('checkpoints/latest.pt').is_file() and Path('checkpoints/best.pt').is_file()
elif 'continuation' in s or any(Path('checkpoints').iterdir()):
    raise RuntimeError('Checkpoint without run progress; inspect transport before restarting')
PY
if [ -f results/train/summary.json ]; then
    /opt/project/.venv/bin/python - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('results/train/summary.json').read_text())
s=json.loads(Path('source.json').read_text())
assert r['complete'] and r['source']==s and r['run_id']==s['run_id'] and r['steps']==s['steps']
for name,meta in r['checkpoint_files'].items():
    assert Path('checkpoints',name).stat().st_size==meta['bytes']
print('Completed run restored; returning existing artifacts')
PY
    exit 0
fi
tar -xf source.tar
tar -xf nanochat-2k-10b-data-7685535.tar
tar -xzf mhc-official-runtime-7546767.tar.gz
export PATH="/opt/project/.venv/bin:$PATH"
export UV_CACHE_DIR="$PWD/uv-cache" UV_LINK_MODE=copy
cp nanochat-src/research/connections/runtime/{pyproject.toml,uv.lock} runtime/
uv sync --frozen --no-build-package transformer-engine-torch --project runtime --python /opt/project/.venv/bin/python
mkdir -p toolchain
cp nanochat-src/research/connections/toolchain/{pyproject.toml,uv.lock} toolchain/
uv sync --frozen --project toolchain --python /opt/project/.venv/bin/python
export PATH="$PWD/runtime/.venv/bin:$PATH"
export CUDA_HOME="$PWD/toolchain/.venv/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/runtime/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
TASK_DRIVER=$(/opt/project/.venv/bin/python - <<'PY'
import ctypes
v=ctypes.c_int();lib=ctypes.CDLL('libcuda.so.1')
assert lib.cuDriverGetVersion(ctypes.byref(v))==0
print(v.value)
PY
)
if [ "$TASK_DRIVER" -lt 13010 ]; then
    dpkg-deb -x cuda-compat-13-1_590.48.01-1_amd64.deb cuda-compat
    export LD_LIBRARY_PATH="$PWD/cuda-compat/usr/local/cuda-13.1/compat:$LD_LIBRARY_PATH"
fi
export NANOCHAT_DTYPE=bfloat16 NANOCHAT_DISABLE_FA3=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export NETRC="$TASK_ROOT/.netrc" WANDB_DIR="$TASK_ROOT/results" WANDB_MODE=online
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none
export NANOCHAT_BASE_DIR="$PWD/cache" OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUBLAS_WORKSPACE_CONFIG=:4096:8
export TRITON_CACHE_DIR="$PWD/triton-cache" TORCHINDUCTOR_CACHE_DIR="$PWD/inductor-cache"
unset NVIDIA_TF32_OVERRIDE
cp source.json results/
cp runtime/uv.lock results/runtime-uv.lock
uv pip freeze --python "$PWD/runtime/.venv/bin/python" > results/environment.txt
cd nanochat-src
cp ../source.json source.json
ln -sfn ../results results
# Validate this source once per GPU family, then start a fresh formal process.
if [[ "${TASK_MOMENTUM_ARGS[0]:-}" == --matrix-optimizer ]]; then
    WANDB_MODE=disabled timeout --kill-after=30s 2400 python research/connections/ngpt/arc_preflight.py
    echo "FORMAL_TRAINING_BEGIN arc_optimizer $TASK_RUN_ID"
else
    if [[ "$TASK_VELOCITY_MODE" == lookahead_axis_write || "$TASK_VELOCITY_MODE" == lookahead_split_write || "$TASK_VELOCITY_MODE" == lookahead_split_learned_beta || "$TASK_VELOCITY_MODE" == lookahead_geodesic_split ]]; then
        if WANDB_MODE=disabled timeout --kill-after=30s 2400 python -m research.connections.ngpt.axis_write_preflight; then
            :
        else
            gate_status=$?
            [[ "${4:-}" == --token-chord-gates || "${4:-}" == --token-beta-gate ]] || exit "$gate_status"
            echo "PREFLIGHT_NEEDS_REPAIR status=$gate_status; retaining allocation for manual retry" >&2
            deadline=$((SECONDS+900))
            while [[ ! -e results/retry-preflight && $SECONDS -lt $deadline ]]; do sleep 2; done
            [[ -e results/retry-preflight ]] || exit "$gate_status"
            rm results/retry-preflight
            WANDB_MODE=disabled timeout --kill-after=30s 2400 python -m research.connections.ngpt.axis_write_preflight
        fi
    else
        WANDB_MODE=disabled python research/connections/ngpt/preflight.py
    fi
    echo "FORMAL_TRAINING_BEGIN nonzero_residual $TASK_RUN_ID"
fi
python -m scripts.connection_train --variant baseline --architecture vanilla --streams 1 \
    --residual-dynamics "$TASK_MODE" --ngpt-velocity-mode "$TASK_VELOCITY_MODE" "${TASK_MOMENTUM_ARGS[@]}" --ngpt-alpha-init .05 --ngpt-force-rms-floor 1e-6 \
    --residual-projection-init normal --residual-projection-init-std .02 --attention-kernel original --attention-backend fa2 \
    --steps "$TASK_STEPS" --schedule-steps "$TASK_SCHEDULE_STEPS" --microbatch 16 --seed 42 \
    --no-activation-checkpoint --no-fused-connection-grad --evaluate-test \
    --data-dir ../data --output-dir results/train --checkpoint-dir ../checkpoints \
    --formal-training --wandb-group "$TASK_GROUP" --run-id "$TASK_RUN_ID" \
    --resume auto --segment-seconds 1800
