#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
mkdir -p results work cache
cp "$0" results/launcher.sh
trap 'tar -cf "$TASK_ROOT/results.tar" -C "$TASK_ROOT" results' EXIT
export PATH="/opt/project/.venv/bin:$PATH"
export NETRC="$PWD/.netrc" WANDB_DIR="$PWD/results"
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none
export NANOCHAT_DTYPE=bfloat16 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export NANOCHAT_BASE_DIR="$PWD/cache"
export OMP_NUM_THREADS=2 RAYON_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUBLAS_WORKSPACE_CONFIG=:4096:8
export TRITON_CACHE_DIR="$PWD/triton-cache" TORCHINDUCTOR_CACHE_DIR="$PWD/inductor-cache"
sha256sum -c inputs.sha256
tar -xf source.tar
tar -xf legacy-baseline.tar
tar -xf data-7444406.tar
tar -xzf cce-3de376c.tar.gz
export PYTHONPATH="$PWD/cce-source${PYTHONPATH:+:$PYTHONPATH}"
export LEGACY_BASELINE_SOURCE="$PWD/legacy-reference/experiment/model.py"
cp source.json results/source.json
cp inputs.sha256 results/inputs.sha256
cp /opt/gradformer-environment.txt results/container-environment.txt
cp /opt/gradformer-locks.txt results/container-locks.txt
cd nanochat-src
ln -s ../results results
cp ../source.json source.json
python - <<'PY'
import torch,triton,cut_cross_entropy
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert torch.__version__.startswith('2.9.1') and triton.__version__=='3.5.1'
assert cut_cross_entropy.__version__=='25.9.3'
print('GPU:',torch.cuda.get_device_name(),flush=True)
PY
python -m scripts.connection_validate
