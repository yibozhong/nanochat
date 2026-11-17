#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
TASK_RUN_ID="$1"
TASK_MODE="$2"
TASK_VELOCITY_MODE="${3:-ambient}"
TASK_MOMENTUM_ARGS=()
TASK_GROUP=residual-nonzero-10b-20261117
case "$TASK_MODE/$TASK_VELOCITY_MODE" in
    ngpt_hbformer/transport_rms)
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
    vanilla/ambient|ngpt/ambient|ngpt_heavy_ball/ambient) ;;
    *) exit 64 ;;
esac
TASK_STEPS=305176
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
cp "$0" results/launcher.sh
/opt/project/.venv/bin/python - "$TASK_RUN_ID" "$TASK_MODE" "$TASK_VELOCITY_MODE" <<'PY'
import json,sys
from pathlib import Path
s=json.loads(Path('source.json').read_text())
assert s['formal_training'] and s['run_id']==sys.argv[1]
assert s['steps']==305176 and s['training_tokens']==10000007168
assert s['initialization']=='seed42_from_scratch_no_diagnostic_weights'
assert s['attention_backend']=='fa2' and s['residual_dynamics']==sys.argv[2]
assert s['model']==dict(depth=8,width=512,heads=4,sequence_length=2048,vocab_size=32768)
assert s['velocity_embedding']==(sys.argv[2] in ('ngpt_heavy_ball','heavy_ball_rms','ngpt_hbformer')) and s['ngpt_alpha_init']==.05
assert s['residual_projection_init']=='normal' and s['residual_projection_init_std']==.02
assert s['ngpt_force_rms_floor']==1e-6
assert s.get('ngpt_velocity_mode','ambient')==sys.argv[3]
if sys.argv[2]=='ngpt_hbformer':
    import math
    assert s['momentum_eta_init']==1 and not s['momentum_eta_trainable'] and s['ngpt_velocity_mode']=='transport_rms'
    assert s['velocity_init']=='token_position' and s['velocity_embedding_std']==.02
    assert s['momentum_beta_init']==.5 and s['momentum_gamma_init']==math.log(2.)
    assert s['backbone_weight_constraint']=='ngpt_unit_l2_init_and_post_step'
elif sys.argv[3]=='transport_rms' or sys.argv[2]=='heavy_ball_rms':
    assert s['momentum_eta_init']==.05 and s['momentum_velocity_rms_floor']==.1
p=Path('results/train/progress.json')
if p.exists():
    r=json.loads(p.read_text())
    assert r['run_id']==s['run_id'] and r['source']==s
    assert Path('checkpoints/latest.pt').is_file() and Path('checkpoints/best.pt').is_file()
elif any(Path('checkpoints').iterdir()):
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
WANDB_MODE=disabled python research/connections/ngpt/preflight.py
echo "FORMAL_TRAINING_BEGIN nonzero_residual $TASK_RUN_ID"
python -m scripts.connection_train --variant baseline --architecture vanilla --streams 1 \
    --residual-dynamics "$TASK_MODE" --ngpt-velocity-mode "$TASK_VELOCITY_MODE" "${TASK_MOMENTUM_ARGS[@]}" --ngpt-alpha-init .05 --ngpt-force-rms-floor 1e-6 \
    --residual-projection-init normal --residual-projection-init-std .02 --attention-kernel original --attention-backend fa2 \
    --steps "$TASK_STEPS" --schedule-steps "$TASK_STEPS" --microbatch 16 --seed 42 \
    --no-activation-checkpoint --no-fused-connection-grad --evaluate-test \
    --data-dir ../data --output-dir results/train --checkpoint-dir ../checkpoints \
    --formal-training --wandb-group "$TASK_GROUP" --run-id "$TASK_RUN_ID" \
    --resume auto --segment-seconds 1800
