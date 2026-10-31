#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
mkdir -p results cache
trap 'tar -cf "$TASK_ROOT/results.tar" -C "$TASK_ROOT" results' EXIT
cp "$0" results/launcher.sh
sha256sum -c inputs.sha256
tar -xf source.tar
tar -xf legacy-source.tar
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
export NANOCHAT_DTYPE=bfloat16 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export NANOCHAT_BASE_DIR="$PWD/cache"
export OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUBLAS_WORKSPACE_CONFIG=:4096:8
export TRITON_CACHE_DIR="$PWD/triton-cache" TORCHINDUCTOR_CACHE_DIR="$PWD/inductor-cache"
export NANOCHAT_CLEANUP_GOLDEN="$PWD/golden"
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
for variant in baseline mhc; do
    python research/connections/nvidia-only/reference.py ../legacy-source "$variant"
done
gate_failed=0
python -m pytest tests/test_nvidia_only_cleanup.py -k parity -v --junitxml=results/controlled.xml || gate_failed=1
cp -r results/cleanup-checks results/controlled-parity
cp -r results/reference-noise results/controlled-noise
unset NVIDIA_TF32_OVERRIDE
python research/connections/nvidia-only/reference.py ../legacy-source mhc
python -m pytest tests/test_mhc_nvidia_full.py -k 'pipeline or model_no' -v --junitxml=results/native.xml || gate_failed=1
python -m pytest tests/test_nvidia_only_cleanup.py -k 'parity and mhc' -v --junitxml=results/native-parity.xml || gate_failed=1
python - <<'PY'
import json,xml.etree.ElementTree as ET
from pathlib import Path
counts={k:0 for k in ['tests','failures','errors','skipped']}
for p in Path('results').glob('*.xml'):
    for suite in ET.parse(p).getroot().findall('testsuite'):
        for key in counts:counts[key]+=int(suite.attrib.get(key,0))
assert not list(Path('results').rglob('*.wandb')) and not (Path('results')/'wandb').exists()
passed=counts['tests']==9 and counts['failures']==counts['errors']==counts['skipped']==0
r=dict(complete=passed,logging_backend='local',checks=counts,prior_passed_checks=dict(job='7564226.0',count=16),source=json.loads(Path('source.json').read_text()),device=json.loads(Path('results/device.json').read_text()),retired_packages_absent=True)
Path('results/cleanup-gate.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
assert passed,counts
PY
exit "$gate_failed"
