#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
mkdir -p results cache
trap 'tar -cf "$TASK_ROOT/results.tar" -C "$TASK_ROOT" results' EXIT
cp "$0" results/launcher.sh
sha256sum -c inputs.sha256
tar -xf source.tar
tar -xf gate-data.tar
tar -xzf mhc-official-runtime-7546767.tar.gz
export PATH="/opt/project/.venv/bin:$PATH"
export UV_CACHE_DIR="$PWD/uv-cache" UV_LINK_MODE=copy
cp nanochat-src/research/connections/runtime/{pyproject.toml,uv.lock} runtime/
uv sync --frozen --no-build-package transformer-engine-torch --project runtime --python /opt/project/.venv/bin/python
cp -r nanochat-src/research/connections/toolchain toolchain
uv sync --frozen --project toolchain --python /opt/project/.venv/bin/python
export PATH="$PWD/runtime/.venv/bin:$PATH"
export CUDA_HOME="$PWD/toolchain/.venv/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/runtime/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export NANOCHAT_DTYPE=bfloat16 PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUBLAS_WORKSPACE_CONFIG=:4096:8
export TRITON_CACHE_DIR="$PWD/triton-cache" TORCHINDUCTOR_CACHE_DIR="$PWD/inductor-cache"
export DG_JIT_CACHE_DIR="$PWD/deepgemm-cache" DG_JIT_USE_NVRTC=0 DG_JIT_CPP_STANDARD=17
export DG_CACHE_DIR="$PWD/deepgemm-cache"
cp source.json inputs.sha256 dependencies.json results/
cp runtime/uv.lock results/runtime-uv.lock
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
mkdir deepgemm
tar -xzf deepgemm.tar.gz --strip-components=1 -C deepgemm
mkdir -p deepgemm/third-party/cutlass deepgemm/third-party/fmt
tar -xzf cutlass.tar.gz --strip-components=1 -C deepgemm/third-party/cutlass
tar -xzf fmt.tar.gz --strip-components=1 -C deepgemm/third-party/fmt
# Install only in this worker's private environment; keep torch/TE/runtime frozen.
uv pip install --python "$TASK_ROOT/runtime/.venv/bin/python" setuptools==80.9.0 wheel==0.45.1 ninja==1.13.0 packaging==25.0
mkdir -p "$TASK_ROOT/cuda-link-libs"
python - <<'PY'
import os
from pathlib import Path
root=Path.cwd()
for name in ('cudart','nvrtc'):
    matches=list((root/'runtime/.venv/lib/python3.11/site-packages/nvidia').rglob('lib'+name+'.so*'))
    matches+=list(Path(os.environ['CUDA_HOME']).rglob('lib'+name+'.so*'))
    assert matches, name
    (root/'cuda-link-libs'/('lib'+name+'.so')).symlink_to(matches[0].resolve())
PY
export LIBRARY_PATH="$TASK_ROOT/cuda-link-libs:${LIBRARY_PATH:-}"
export DG_FORCE_BUILD=1
cd deepgemm
python setup.py bdist_wheel > "$TASK_ROOT/results/deepgemm-build.log" 2>&1
uv pip install --python "$TASK_ROOT/runtime/.venv/bin/python" --no-deps dist/*.whl
cd "$TASK_ROOT"
uv pip freeze --python "$TASK_ROOT/runtime/.venv/bin/python" > results/environment.txt
unset NVIDIA_TF32_OVERRIDE
cd nanochat-src
ln -sfn ../results results
cp ../source.json source.json
python research/connections/hash-kernel-bench/bench.py --data ../data/train.bin --output results/hash-kernel-bench
