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
export NANOCHAT_DTYPE=bfloat16 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
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
export NVIDIA_TF32_OVERRIDE=0
# The existing nanochat Hopper FA3 distribution, pinned and fetched once.
export NANOCHAT_FA3_REVISION=de87b9b5af06dd9984df595bef90b2eba44b181a
export HF_HOME="$TASK_ROOT/fa3-cache"
HF_HUB_OFFLINE=0 python - <<'FA3PY'
import hashlib,json,os
from pathlib import Path
from kernels import get_kernel
revision=os.environ['NANOCHAT_FA3_REVISION']
module=get_kernel('varunneal/flash-attention-3',revision=revision)
interface=module.flash_attn_interface
folder=Path(interface.__file__).parent
files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
       for p in folder.iterdir() if p.is_file()}
Path('results/fa3-artifacts.json').write_text(json.dumps(dict(repo='varunneal/flash-attention-3',revision=revision,files=files),indent=2)+'\n')
print('Pinned FA3 loaded:',interface.__file__,flush=True)
FA3PY
python -m scripts.connection_tokenres_gate
