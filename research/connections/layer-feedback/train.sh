#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
TASK_RUN_ID="$1"
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
/opt/project/.venv/bin/python - "$TASK_RUN_ID" <<'PY'
import json,sys
from pathlib import Path
s=json.loads(Path('source.json').read_text())
assert s['formal_training'] and s['run_id']==sys.argv[1]
assert s['steps']==305176 and s['training_tokens']==10000007168
assert s['initialization']=='seed42_from_zero_no_diagnostic_weights'
assert s['attention_backend']=='fa2' and s['pass_mix']==[.15,.85] and s['model_variant']=='layer_feedback'
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
# Validate the new layer-attention model before any formal weights are created.
# Successful per-source/family markers survive checkpoint transfers.
WANDB_MODE=disabled python research/connections/layer-feedback/preflight.py
echo "FORMAL_TRAINING_BEGIN layer_feedback $TASK_RUN_ID"
python -m scripts.connection_train --variant baseline --architecture vanilla --streams 1 \
    --latent-feedback --feedback-source layers --feedback-attention-backend triton --feedback-eval-passes 4 --attention-kernel original --attention-backend fa2 \
    --steps "$TASK_STEPS" --schedule-steps "$TASK_STEPS" --microbatch 16 --seed 42 \
    --no-activation-checkpoint --no-fused-connection-grad --evaluate-test \
    --data-dir ../data --output-dir results/train --checkpoint-dir ../checkpoints \
    --formal-training --wandb-group layer-feedback-10b-20261113 --run-id "$TASK_RUN_ID" \
    --resume auto --segment-seconds 1800
