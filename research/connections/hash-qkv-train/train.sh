#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
TASK_VARIANT=hash_qkv
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
mkdir -p work/preflight-data
tar -xf gate-data.tar -C work/preflight-data
tar -xzf mhc-official-runtime-7546767.tar.gz
export PATH="/opt/project/.venv/bin:$PATH"
export UV_CACHE_DIR="$PWD/uv-cache" UV_LINK_MODE=copy
cp nanochat-src/research/connections/runtime/{pyproject.toml,uv.lock} runtime/
uv sync --frozen --no-build-package transformer-engine-torch --project runtime --python /opt/project/.venv/bin/python
cp -r nanochat-src/research/connections/toolchain toolchain
uv sync --frozen --project toolchain --python /opt/project/.venv/bin/python
export PATH="$PWD/runtime/.venv/bin:$PATH"
export CUDA_HOME="$PWD/toolchain/.venv/lib/python3.11/site-packages/nvidia/cu13"
mkdir cuda-binary-tools
tar -xJf cuobjdump.tar.xz --strip-components=1 -C cuda-binary-tools
ln -s "$TASK_ROOT/cuda-binary-tools/bin/cuobjdump" "$CUDA_HOME/bin/cuobjdump"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/runtime/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export NANOCHAT_DTYPE=bfloat16 PYTHONUNBUFFERED=1 WANDB_MODE=online HF_HUB_OFFLINE=1
export NETRC="$TASK_ROOT/.netrc" WANDB_DIR="$TASK_ROOT/results"
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
mkdir deepgemm
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
TASK_GPU_FAMILY=$(python - <<'HOPPER'
import json,torch
from pathlib import Path
s=json.loads(Path('source.json').read_text())
assert s['formal_training'] and s['model_variant']=='hash_qkv'
assert s['initialization']=='seed42_from_zero_no_validation_weights'
assert torch.cuda.get_device_capability()==(9,0)
name=torch.cuda.get_device_name()
assert 'H100' in name or 'H200' in name
print('H100' if 'H100' in name else 'H200')
HOPPER
)
# First use of each Hopper family validates the new training interface and resume.
# It writes separate diagnostic checkpoints and deletes them before formal init.
WANDB_MODE=disabled python -m scripts.hash_qkv_preflight --family "$TASK_GPU_FAMILY"
echo "FORMAL_TRAINING_BEGIN $TASK_VARIANT $TASK_RUN_ID"
python -m scripts.hash_qkv_train --attention-backend hardware_auto \
    --steps "$TASK_STEPS" --schedule-steps "$TASK_STEPS" --microbatch "$TASK_MICROBATCH" --seed 42 \
    --no-activation-checkpoint --evaluate-test \
    --data-dir ../data --output-dir results/train --checkpoint-dir ../checkpoints \
    --formal-training --wandb-group hash-qkv-10b-20261107 --run-id "$TASK_RUN_ID" \
    --resume auto --segment-seconds 1800
