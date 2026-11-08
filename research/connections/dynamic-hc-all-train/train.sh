#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
TASK_VARIANT=dynamic_hc_all
TASK_RUN_ID="$1"
TASK_STEPS=305176
TASK_MICROBATCH="$2"
mkdir -p checkpoints
if [ -f results.tar ]; then tar -xf results.tar; fi
if [ -f checkpoints.tar ]; then tar -xf checkpoints.tar; fi
mkdir -p results work cache
cp "$0" results/launcher.sh
# Only fully written archives are eligible for a scheduler checkpoint transfer.
checkpoint_archives_on_exit() {
    TASK_EXIT_STATUS=$?
    trap - EXIT
    if ! tar -cf "$TASK_ROOT/checkpoints.tar.tmp" -C "$TASK_ROOT" checkpoints ||
       ! mv "$TASK_ROOT/checkpoints.tar.tmp" "$TASK_ROOT/checkpoints.tar" ||
       ! tar -cf "$TASK_ROOT/results.tar.tmp" -C "$TASK_ROOT" results ||
       ! mv "$TASK_ROOT/results.tar.tmp" "$TASK_ROOT/results.tar"; then
        # Do not report a successful checkpoint (85) after an archive failure.
        exit 74
    fi
    exit "$TASK_EXIT_STATUS"
}
trap checkpoint_archives_on_exit EXIT
sha256sum -c inputs.sha256
# A transferred completion marker must never trigger training from step zero.
if [ -f results/train/summary.json ]; then
    /opt/project/.venv/bin/python - "$TASK_VARIANT" "$TASK_RUN_ID" "$TASK_STEPS" <<'PYDONE'
import json,sys
from pathlib import Path
r=json.loads(Path('results/train/summary.json').read_text())
s=json.loads(Path('source.json').read_text())
assert r['complete'] and r['source']==s and r['variant']==sys.argv[1]
assert r['run_id']==sys.argv[2] and r['steps']==int(sys.argv[3])
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
export NANOCHAT_DTYPE=bfloat16 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 WANDB_MODE=online
export NETRC="$TASK_ROOT/.netrc" WANDB_DIR="$TASK_ROOT/results"
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none
export NANOCHAT_BASE_DIR="$PWD/cache"
export OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUBLAS_WORKSPACE_CONFIG=:4096:8
export TRITON_CACHE_DIR="$PWD/triton-cache" TORCHINDUCTOR_CACHE_DIR="$PWD/inductor-cache"
cp source.json inputs.sha256 results/
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
python nanochat-src/research/connections/attnres/device_probe.py
source nanochat-src/research/connections/runtime/hardware-device.sh
cd nanochat-src
ln -sfn ../results results
cp ../source.json source.json
python - <<'PY'
import importlib.util,json,torch
from pathlib import Path
assert importlib.util.find_spec('tile_kernels') is None
assert importlib.util.find_spec('tilelang') is None
assert torch.cuda.is_available() and torch.cuda.device_count()==1
Path('results/device.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),torch=torch.__version__),indent=2)+'\n')
print('NVIDIA-only runtime verified; GPU:',torch.cuda.get_device_name(),flush=True)
PY
unset NVIDIA_TF32_OVERRIDE
# Match the published gate execution hashes before any formal model is created.
python - <<'VALIDATED'
import json,hashlib
from pathlib import Path
s=json.loads(Path('source.json').read_text())
assert s['formal_training'] and s['model_variant']=='dynamic_hc_all'
assert s['initialization']=='seed42_from_zero_no_validation_weights'
assert s['steps']==305176 and s['training_tokens']==10000007168
for name,digest in s['validated_execution_files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256(Path('research/connections/dynamic-hc-all-train/preflight.py').read_bytes()).hexdigest()==s['preflight_sha256']
VALIDATED
# Runs once per source/run. Checkpoint transfers retain the successful marker.
# No W&B and no preflight weights are passed to the formal training directory.
WANDB_MODE=disabled python research/connections/dynamic-hc-all-train/preflight.py --microbatch "$TASK_MICROBATCH"
echo "FORMAL_TRAINING_BEGIN $TASK_VARIANT $TASK_RUN_ID"
python -m scripts.connection_train --architecture vanilla --variant dynamic_hc_all --streams 4 --matched-backbone-init \
    --sequence-length 2048 --attention-kernel original --attention-backend hardware_auto \
    --steps "$TASK_STEPS" --schedule-steps "$TASK_STEPS" --microbatch "$TASK_MICROBATCH" --seed 42 \
    --no-activation-checkpoint --no-fused-connection-grad --evaluate-test \
    --data-dir ../data --output-dir results/train --checkpoint-dir ../checkpoints \
    --formal-training --wandb-group dynamic-hc-all-10b-20261108 --run-id "$TASK_RUN_ID" \
    --resume auto --segment-seconds 1800
