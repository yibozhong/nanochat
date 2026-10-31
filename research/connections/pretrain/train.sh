#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
TASK_VARIANT="$1"
TASK_RUN_ID="$2"
TASK_STEPS="$3"
shift 3
mkdir -p checkpoints
if [ -f results.tar ]; then tar -xf results.tar; fi
if [ -f checkpoints.tar ]; then tar -xf checkpoints.tar; fi
mkdir -p results work cache
cp "$0" results/launcher.sh
trap 'tar -cf "$TASK_ROOT/results.tar" -C "$TASK_ROOT" results; tar -cf "$TASK_ROOT/checkpoints.tar" -C "$TASK_ROOT" checkpoints' EXIT
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
tar -xf legacy-baseline.tar
tar -xf "${NANOCHAT_DATA_ARCHIVE:-data-7444406.tar}"
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
cp nanochat-src/research/connections/nvidia-native/runtime/{pyproject.toml,uv.lock} runtime/
uv sync --frozen --no-build-package transformer-engine-torch --project runtime --python /opt/project/.venv/bin/python 2>&1 | tee results/nvidia-install.log
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
ln -sfn ../results results
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
if [ "$TASK_VARIANT" = mhc ]; then
    TASK_BACKEND=nvidia
    unset NVIDIA_TF32_OVERRIDE
else
    TASK_BACKEND=deepseek
    export NVIDIA_TF32_OVERRIDE=0
fi
python -m scripts.connection_train --variant "$TASK_VARIANT" --connection-backend "$TASK_BACKEND" \
    --steps "$TASK_STEPS" --schedule-steps "$TASK_STEPS" --microbatch 16 \
    --attention-backend sdpa_cudnn --matched-backbone-init --evaluate-test \
    --no-activation-checkpoint --no-fused-connection-grad --seed 42 \
    --data-dir ../data --output-dir results/train --checkpoint-dir ../checkpoints \
    --run-id "$TASK_RUN_ID" --formal-training --wandb-group mhc-pretrain-20261030 --resume auto --segment-seconds 14400
