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
export PATH="$PWD/runtime/.venv/bin:/opt/project/.venv/bin:$PATH"
export CUDA_HOME="$PWD/runtime/.venv/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export NETRC="$PWD/.netrc" WANDB_DIR="$PWD/results"
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none
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
uv pip freeze --python "$PWD/runtime/.venv/bin/python" > results/environment.txt
nvcc --version > results/nvcc.txt
cd nanochat-src
ln -s ../results results
cp ../source.json source.json
python - <<'PYCHECK'
import torch,tilelang,cut_cross_entropy
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert torch.__version__.startswith('2.10.0') and tilelang.__version__=='0.1.9'
assert torch.cuda.get_device_capability()[0] in (9,10)
assert cut_cross_entropy.__version__=='25.9.3'
print('GPU:',torch.cuda.get_device_name(),torch.__version__,torch.version.cuda,flush=True)
PYCHECK
python -m scripts.connection_validate
