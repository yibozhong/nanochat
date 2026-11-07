#!/bin/bash
# Sourced from the task scratch root; safe on scheduler restart in the same directory.
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
mkdir -p cuda-binary-tools
tar -xJf cuobjdump.tar.xz --strip-components=1 -C cuda-binary-tools
ln -sfn "$TASK_ROOT/cuda-binary-tools/bin/cuobjdump" "$CUDA_HOME/bin/cuobjdump"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/runtime/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export NANOCHAT_DTYPE=bfloat16 PYTHONUNBUFFERED=1 WANDB_MODE="${WANDB_MODE:-disabled}" HF_HUB_OFFLINE=1
export WANDB_DIR="$TASK_ROOT/results"
if [ -f "$TASK_ROOT/.netrc" ]; then export NETRC="$TASK_ROOT/.netrc"; fi
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none
export NANOCHAT_BASE_DIR="$PWD/cache"
export OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2
# 4096-expert FFNs exceed TE's default 20 MiB host tensor-handle pool.
export NVTE_TENSOR_HANDLE_POOL_SIZE_MB=128
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUBLAS_WORKSPACE_CONFIG=:4096:8
export TRITON_CACHE_DIR="$PWD/triton-cache" TORCHINDUCTOR_CACHE_DIR="$PWD/inductor-cache"
# DeepGEMM's host extension uses C++17, but its NVCC JIT defaults to C++20.
# Keep that default: its host -fconcepts flag is incompatible with a C++17
# device parser when GCC 12 standard-library headers expose concepts.
export DG_JIT_CACHE_DIR="$PWD/deepgemm-cache" DG_JIT_USE_NVRTC=0 DG_JIT_CPP_STANDARD=20
export DG_JIT_PRINT_COMPILER_COMMAND=1
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
source nanochat-src/research/connections/runtime/hardware-device.sh
mkdir -p deepgemm
tar -xzf deepgemm.tar.gz --strip-components=1 -C deepgemm
mkdir -p deepgemm/third-party/cutlass deepgemm/third-party/fmt
tar -xzf cutlass.tar.gz --strip-components=1 -C deepgemm/third-party/cutlass
tar -xzf fmt.tar.gz --strip-components=1 -C deepgemm/third-party/fmt
# Install only in this worker's private environment; keep torch/TE/runtime frozen.
uv pip install --python "$TASK_ROOT/runtime/.venv/bin/python" setuptools==80.9.0 wheel==0.45.1 ninja==1.13.0 packaging==25.0
# CUDA_HOME contains the compiler/core SDK; PyTorch's math-library headers live
# in the separate, already pinned runtime wheels. Expose both to host/JIT builds.
HASH_CUDA_HEADERS=$(python - <<'PY'
import json
import os
from pathlib import Path
root = Path.cwd()
sdk = Path(os.environ['CUDA_HOME'])
runtime = root/'runtime/.venv/lib/python3.11/site-packages/nvidia'
includes = [sdk/'include', sdk/'include/cccl']
includes += sorted(p for p in runtime.rglob('include') if p.is_dir())
includes = list(dict.fromkeys(str(p.resolve()) for p in includes if p.is_dir()))
required = ('cusparse.h', 'cublas_v2.h', 'cublasLt.h', 'cusolverDn.h', 'nvrtc.h')
headers = {name: next((str(Path(p)/name) for p in includes if (Path(p)/name).is_file()), None)
           for name in required}
(root/'results/build-headers.json').write_text(json.dumps(dict(include_paths=includes, headers=headers), indent=2)+'\n')
assert all(headers.values()), headers
print(':'.join(includes))
PY
)
export CPATH="$HASH_CUDA_HEADERS${CPATH:+:$CPATH}"
# Exercise both external commands used by DeepGEMM before building its extension.
python - <<'PY'
import json
import os
from pathlib import Path
import subprocess
root = Path.cwd()
sdk = Path(os.environ['CUDA_HOME'])
report = {}
for name in ('nvcc', 'cuobjdump'):
    report[name] = subprocess.check_output([str(sdk/'bin'/name), '--version'], text=True)
source = root/'cache/toolchain-probe.cu'
cubin = root/'cache/toolchain-probe.cubin'
source.write_text('#include <string>\nextern "C" __global__ void hash_toolchain_probe(float* x) { if (threadIdx.x == 0) *x += 1; }\n')
subprocess.run([str(sdk/'bin/nvcc'), str(source), '-std=c++20', '--compiler-options=-fconcepts', '-arch=sm_90a', '-cubin', '-o', str(cubin)], check=True)
report['symbols'] = subprocess.check_output([str(sdk/'bin/cuobjdump'), '-symbols', str(cubin)], text=True)
assert 'hash_toolchain_probe' in report['symbols']
(root/'results/toolchain-probe.json').write_text(json.dumps(report, indent=2)+'\n')
source.unlink(); cubin.unlink()
PY
mkdir -p "$TASK_ROOT/cuda-link-libs"
python - <<'PY'
import os
from pathlib import Path
root=Path.cwd()
for name in ('cudart','nvrtc'):
    matches=list((root/'runtime/.venv/lib/python3.11/site-packages/nvidia').rglob('lib'+name+'.so*'))
    matches+=list(Path(os.environ['CUDA_HOME']).rglob('lib'+name+'.so*'))
    assert matches, name
    link=root/'cuda-link-libs'/('lib'+name+'.so')
    link.unlink(missing_ok=True)
    link.symlink_to(matches[0].resolve())
PY
export LIBRARY_PATH="$TASK_ROOT/cuda-link-libs:${LIBRARY_PATH:-}"
export DG_FORCE_BUILD=1
cd deepgemm
python setup.py bdist_wheel > "$TASK_ROOT/results/deepgemm-build.log" 2>&1
uv pip install --python "$TASK_ROOT/runtime/.venv/bin/python" --no-deps dist/*.whl
cd "$TASK_ROOT"
uv pip freeze --python "$TASK_ROOT/runtime/.venv/bin/python" > results/environment.txt
unset NVIDIA_TF32_OVERRIDE
