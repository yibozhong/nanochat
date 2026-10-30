#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
mkdir -p results work cache
cp "$0" results/launcher.sh
trap 'tar -cf "$TASK_ROOT/results.tar" -C "$TASK_ROOT" results' EXIT
sha256sum -c inputs.sha256
tar -xf source.tar
tar -xf legacy-baseline.tar
tar -xf data-7444406.tar
tar -xzf mhc-official-runtime-7546767.tar.gz
export PATH="/opt/project/.venv/bin:$PATH"
export UV_CACHE_DIR="$PWD/uv-cache" UV_LINK_MODE=copy
cp nanochat-src/research/connections/runtime/{pyproject.toml,uv.lock} runtime/
uv sync --frozen --project runtime --python /opt/project/.venv/bin/python
cp -r nanochat-src/research/connections/toolchain toolchain
uv sync --frozen --project toolchain --python /opt/project/.venv/bin/python
export PATH="$PWD/runtime/.venv/bin:$PATH"
export CUDA_HOME="$PWD/toolchain/.venv/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/runtime/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export MAX_JOBS=2 NVTE_BUILD_THREADS_PER_JOB=1 NVTE_WITH_NCCL_EP=0
# Torch ships library headers separately from the isolated nvcc toolchain.
export CPATH="$CUDA_HOME/include:$PWD/runtime/.venv/lib/python3.11/site-packages/nvidia/cu13/include:$PWD/runtime/.venv/lib/python3.11/site-packages/nvidia/nccl/include:$PWD/runtime/.venv/lib/python3.11/site-packages/nvidia/cudnn/include"
cp nanochat-src/research/connections/nvidia-projection/runtime/{pyproject.toml,uv.lock} runtime/
uv sync --frozen --project runtime --python /opt/project/.venv/bin/python 2>&1 | tee results/nvidia-install.log
export NETRC="$PWD/.netrc" WANDB_DIR="$PWD/results"
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none
export NANOCHAT_MHC_TEST_PORTABILITY=1
export NANOCHAT_DTYPE=bfloat16 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export NANOCHAT_BASE_DIR="$PWD/cache"
export OMP_NUM_THREADS=2 RAYON_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUBLAS_WORKSPACE_CONFIG=:4096:8
export TRITON_CACHE_DIR="$PWD/triton-cache" TORCHINDUCTOR_CACHE_DIR="$PWD/inductor-cache"
export TILELANG_CACHE_DIR="$PWD/tilelang-cache"
export LEGACY_BASELINE_SOURCE="$PWD/legacy-reference/experiment/model.py"
cp source.json results/source.json
cp inputs.sha256 results/inputs.sha256
cp runtime/uv.lock results/runtime-uv.lock
cp toolchain/uv.lock results/toolchain-uv.lock
python nanochat-src/scripts/cache_te_extension.py
uv pip freeze --python "$PWD/runtime/.venv/bin/python" > results/environment.txt
uv pip freeze --python "$PWD/toolchain/.venv/bin/python" > results/toolchain-environment.txt
nvcc --version > results/nvcc.txt
SYSTEM_CUDA_DRIVER=$(/opt/project/.venv/bin/python - <<'PYDRIVER'
import ctypes
v=ctypes.c_int()
lib=ctypes.CDLL('libcuda.so.1')
assert lib.cuDriverGetVersion(ctypes.byref(v))==0
print(v.value)
PYDRIVER
)
if [ "$SYSTEM_CUDA_DRIVER" -lt 13010 ]; then
    dpkg-deb -x cuda-compat-13-1_590.48.01-1_amd64.deb cuda-compat
    export LD_LIBRARY_PATH="$PWD/cuda-compat/usr/local/cuda-13.1/compat:$LD_LIBRARY_PATH"
    echo "NVIDIA cuda-compat-13-1 for system CUDA driver $SYSTEM_CUDA_DRIVER" > results/compatibility.txt
else
    echo "Native system CUDA driver $SYSTEM_CUDA_DRIVER" > results/compatibility.txt
fi
cd nanochat-src
ln -s ../results results
cp ../source.json source.json
python - <<'PYCHECK'
import torch,tilelang,cut_cross_entropy
from nanochat.engine import KVCache
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert torch.__version__.startswith('2.10.0') and tilelang.__version__=='0.1.9'
assert torch.cuda.get_device_capability()[0] >= 8
assert cut_cross_entropy.__version__=='25.9.3'
print('GPU:',torch.cuda.get_device_name(),torch.__version__,torch.version.cuda,flush=True)
PYCHECK
python -m "${NANOCHAT_CONNECTION_ENTRYPOINT:-scripts.connection_validate}" "$@"
