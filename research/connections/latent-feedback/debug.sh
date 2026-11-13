#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
mkdir -p results
finish() {
    TASK_STATUS=$?
    trap - EXIT
    if [ "$TASK_STATUS" -ne 0 ]; then
        echo "DEBUG_FAILED status=$TASK_STATUS; keeping allocation for 900 seconds" >&2
        sleep 900
    fi
    tar -cf "$TASK_ROOT/results.tar.tmp" -C "$TASK_ROOT" results || exit 74
    mv "$TASK_ROOT/results.tar.tmp" "$TASK_ROOT/results.tar" || exit 74
    exit "$TASK_STATUS"
}
trap finish EXIT
tar -xf source.tar
tar -xf gate-data.tar
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
export NANOCHAT_DTYPE=bfloat16 NANOCHAT_DISABLE_FA3=1 HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export NANOCHAT_BASE_DIR="$PWD/cache" OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUBLAS_WORKSPACE_CONFIG=:4096:8
export TRITON_CACHE_DIR="$PWD/triton-cache" TORCHINDUCTOR_CACHE_DIR="$PWD/inductor-cache"
unset NVIDIA_TF32_OVERRIDE
cp source.json results/
cp runtime/uv.lock results/runtime-uv.lock
uv pip freeze --python "$PWD/runtime/.venv/bin/python" > results/environment.txt
cd nanochat-src
cp ../source.json source.json
ln -s ../results results
timeout --signal=TERM --kill-after=30s 2400 bash research/connections/latent-feedback/check.sh
