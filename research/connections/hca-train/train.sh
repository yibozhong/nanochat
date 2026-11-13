#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
TASK_MODE="$1"
TASK_RUN_ID="$2"
mkdir -p checkpoints results
if [ -f results.tar ]; then tar -xf results.tar; fi
if [ -f checkpoints.tar ]; then tar -xf checkpoints.tar; fi
checkpoint_archives_on_exit() {
    TASK_EXIT_STATUS=$?
    trap - EXIT
    printf '%s\n' "$TASK_EXIT_STATUS" > "$TASK_ROOT/results/exit-code.txt"
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
if [ -f results/train/summary.json ]; then
    /opt/project/.venv/bin/python - "$TASK_MODE" "$TASK_RUN_ID" <<'PYDONE'
import json,sys
from pathlib import Path
r=json.loads(Path('results/train/summary.json').read_text())
s=json.loads(Path('source.json').read_text())
assert r['complete'] and r['source']==s and s['attention_mode']==sys.argv[1]
assert r['run_id']==sys.argv[2] and r['steps']==305176 and r['training_tokens']==10000007168
for name,meta in r['checkpoint_files'].items():
    assert Path('checkpoints',name).stat().st_size==meta['bytes']
print('Completed run restored; returning existing artifacts without more training')
PYDONE
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
export PYTHONPATH="$TASK_ROOT/nanochat-src${PYTHONPATH:+:$PYTHONPATH}"
export NANOCHAT_DTYPE=bfloat16 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export NETRC="$TASK_ROOT/.netrc" WANDB_DIR="$TASK_ROOT/results"
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none
export NANOCHAT_BASE_DIR="$PWD/cache"
export OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUBLAS_WORKSPACE_CONFIG=:4096:8
export TRITON_CACHE_DIR="$PWD/triton-cache" TORCHINDUCTOR_CACHE_DIR="$PWD/inductor-cache"
unset NVIDIA_TF32_OVERRIDE
mkdir -p cache
cp source.json results/
cp runtime/uv.lock results/runtime-uv.lock
uv pip freeze --python "$PWD/runtime/.venv/bin/python" > results/environment.txt
SYSTEM_CUDA_DRIVER=$(/opt/project/.venv/bin/python - <<'PY'
import ctypes
v=ctypes.c_int();lib=ctypes.CDLL('libcuda.so.1')
assert lib.cuDriverGetVersion(ctypes.byref(v))==0
print(v.value)
PY
)
if [ "$SYSTEM_CUDA_DRIVER" -lt 13010 ]; then
    dpkg-deb -x cuda-compat-13-1_590.48.01-1_amd64.deb cuda-compat
    export LD_LIBRARY_PATH="$PWD/cuda-compat/usr/local/cuda-13.1/compat:$LD_LIBRARY_PATH"
fi
cd nanochat-src
ln -sfn ../results results
cp ../source.json source.json
# Successful startup persists across exit85. No diagnostic model/RNG/data
# state enters the separate formal training process or checkpoints directory.
if timeout --signal=TERM --kill-after=30 3600 python research/connections/hca-train/preflight.py --mode "$TASK_MODE"; then
    TASK_MICROBATCH=$(python -c 'import json;print(json.load(open("results/hca-startup/passed.json"))["microbatch"])')
else
    failure=$?
    echo "HCA_PREFLIGHT_FAILED $TASK_MODE $TASK_RUN_ID exit=$failure; preserving allocation for 600 seconds; no automatic retry."
    sleep 600
    exit "$failure"
fi
export WANDB_MODE=online
echo "FORMAL_TRAINING_BEGIN $TASK_MODE $TASK_RUN_ID"
python -m scripts.connection_train --architecture vanilla --variant baseline --streams 1 \
    --sequence-length 2048 --attention-kernel original --attention-mode "$TASK_MODE" \
    --hca-block-size 16 --hca-window 256 --hca-ratio-loss-weight 1 --hca-temperature 1 --hca-noise \
    --steps 305176 --schedule-steps 305176 --microbatch "$TASK_MICROBATCH" --seed 42 \
    --no-activation-checkpoint --no-fused-connection-grad --evaluate-test \
    --data-dir ../data --output-dir results/train --checkpoint-dir ../checkpoints \
    --formal-training --wandb-group hca-10b-20261112 --run-id "$TASK_RUN_ID" \
    --resume auto --segment-seconds 1800
