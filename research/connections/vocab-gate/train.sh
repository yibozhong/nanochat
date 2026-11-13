#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
TASK_RUN_ID="$1"
mkdir -p checkpoints
if [ -f results.tar ]; then tar -xf results.tar; fi
if [ -f checkpoints.tar ]; then tar -xf checkpoints.tar; fi
mkdir -p results cache
cp "$0" results/launcher.sh
# Same atomic archive/exit85 contract as the verified scheduler transport.
finish() {
    TASK_STATUS=$?
    trap - EXIT
    if [ "$TASK_STATUS" -ne 0 ] && [ "$TASK_STATUS" -ne 85 ]; then
        echo "VOCAB_GATE_FAILED $TASK_RUN_ID exit=$TASK_STATUS; allocation retained 600 seconds; no automatic retry."
        sleep 600
    fi
    if ! tar -cf "$TASK_ROOT/checkpoints.tar.tmp" -C "$TASK_ROOT" checkpoints ||
       ! mv "$TASK_ROOT/checkpoints.tar.tmp" "$TASK_ROOT/checkpoints.tar" ||
       ! tar -cf "$TASK_ROOT/results.tar.tmp" -C "$TASK_ROOT" results ||
       ! mv "$TASK_ROOT/results.tar.tmp" "$TASK_ROOT/results.tar"; then
        exit 74
    fi
    exit "$TASK_STATUS"
}
trap finish EXIT
rm -f results/complete-restored
tar -xf source.tar
export PATH="/opt/project/.venv/bin:$PATH"
python - "$TASK_RUN_ID" <<'PY'
import sys,json
from pathlib import Path
sys.path.insert(0,'nanochat-src/research/connections/vocab-gate')
from run_policy import validate_source,validate_complete
s=json.loads(Path('source.json').read_text())
validate_source(s)
assert s['run_id']==sys.argv[1]
assert Path('nanochat-2k-10b-data-7685535.tar').stat().st_size==s['data_archive_bytes']
if Path('results/train/progress.json').exists():
    p=json.loads(Path('results/train/progress.json').read_text())
    assert p['run_id']==s['run_id'] and p['source']==s
    assert Path('checkpoints/latest.pt').is_file() and Path('checkpoints/best.pt').is_file(), 'Missing persisted checkpoint; refusing to restart from zero'
if Path('results/train/summary.json').exists():
    r=json.loads(Path('results/train/summary.json').read_text());validate_complete(r,s)
    for name,meta in r['checkpoint_files'].items():
        assert Path('checkpoints',name).stat().st_size==meta['bytes']
    Path('results/complete-restored').touch()
PY
if [ -f results/complete-restored ]; then exit 0; fi
tar -xf nanochat-2k-10b-data-7685535.tar
tar -xzf mhc-official-runtime-7546767.tar.gz
export UV_CACHE_DIR="$TASK_ROOT/uv-cache" UV_LINK_MODE=copy
cp nanochat-src/research/connections/runtime/{pyproject.toml,uv.lock} runtime/
timeout --signal=TERM --kill-after=30 900 uv sync --frozen --no-build-package transformer-engine-torch --project runtime --python /opt/project/.venv/bin/python
mkdir -p toolchain
cp nanochat-src/research/connections/toolchain/{pyproject.toml,uv.lock} toolchain/
timeout --signal=TERM --kill-after=30 900 uv sync --frozen --project toolchain --python /opt/project/.venv/bin/python
export PATH="$TASK_ROOT/runtime/.venv/bin:$PATH"
export CUDA_HOME="$TASK_ROOT/toolchain/.venv/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$TASK_ROOT/runtime/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TASK_ROOT/nanochat-src"
export NANOCHAT_DTYPE=bfloat16 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 NANOCHAT_DISABLE_FA3=1
export NANOCHAT_BASE_DIR="$TASK_ROOT/cache"
export OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUBLAS_WORKSPACE_CONFIG=:4096:8
export TRITON_CACHE_DIR="$TASK_ROOT/triton-cache" TORCHINDUCTOR_CACHE_DIR="$TASK_ROOT/inductor-cache"
export NETRC="$TASK_ROOT/.netrc" WANDB_DIR="$TASK_ROOT/results" WANDB_MODE=disabled
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none
unset NVIDIA_TF32_OVERRIDE
SYSTEM_CUDA_DRIVER=$(/opt/project/.venv/bin/python - <<'PY'
import ctypes
v=ctypes.c_int();lib=ctypes.CDLL('libcuda.so.1')
assert lib.cuDriverGetVersion(ctypes.byref(v))==0
print(v.value)
PY
)
if [ "$SYSTEM_CUDA_DRIVER" -lt 13010 ]; then
    dpkg-deb -x cuda-compat-13-1_590.48.01-1_amd64.deb cuda-compat
    export LD_LIBRARY_PATH="$TASK_ROOT/cuda-compat/usr/local/cuda-13.1/compat:$LD_LIBRARY_PATH"
fi
cp source.json results/source.json
cp runtime/uv.lock results/runtime-uv.lock
cp toolchain/uv.lock results/toolchain-uv.lock
uv pip freeze --python "$TASK_ROOT/runtime/.venv/bin/python" > results/environment.txt
cd nanochat-src
ln -sfn ../results results
cp ../source.json source.json
python - <<'PY'
import json,torch,triton
from pathlib import Path
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert torch.cuda.get_device_capability()==(9,0)
assert any(name in torch.cuda.get_device_name() for name in ('H100','H200'))
assert torch.__version__=='2.10.0+cu130' and triton.__version__=='3.6.0'
Path('results/device.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
    capability=torch.cuda.get_device_capability(),torch=torch.__version__,triton=triton.__version__,
    attention_backend='fa2',validation='same-allocation before formal'),indent=2)+'\n')
PY
if [ ! -f results/gate-approved.json ]; then
    mkdir -p ../requests results/debug-requests
    cp research/connections/vocab-gate/check.sh ../requests/01-check.sh
    echo "VOCAB_GATE_GPU_READY $TASK_RUN_ID"
    TASK_DEBUG_START=$(date +%s)
    while [ ! -f results/gate-approved.json ]; do
        for TASK_REQUEST in ../requests/*.sh; do
            [ -f "$TASK_REQUEST" ] || continue
            TASK_NAME=${TASK_REQUEST##*/}
            [ ! -f "results/debug-requests/$TASK_NAME.exit" ] || continue
            set +e
            timeout --signal=TERM --kill-after=30 2400 bash "$TASK_REQUEST" > "results/debug-requests/$TASK_NAME.log" 2>&1
            TASK_RC=$?
            set -e
            printf '%s\n' "$TASK_RC" > "results/debug-requests/$TASK_NAME.exit"
            echo "VOCAB_GATE_REQUEST_DONE $TASK_NAME $TASK_RC"
        done
        if (( $(date +%s) - TASK_DEBUG_START > 10800 )); then exit 124; fi
        sleep 5
    done
fi
python - <<'PYGATE'
import json,sys
from pathlib import Path
sys.path.insert(0,'research/connections/vocab-gate')
from run_policy import validate_gate
s=json.loads(Path('source.json').read_text())
validate_gate(json.loads(Path('results/gate-approved.json').read_text()),s)
PYGATE
export WANDB_MODE=online
echo "FORMAL_TRAINING_BEGIN vocab_gate_dense $TASK_RUN_ID"
python -m scripts.connection_train --architecture vanilla --variant baseline --streams 1 \
    --sequence-length 2048 --attention-kernel original --attention-backend fa2 \
    --vocab-gate --vocab-gate-alpha .5 --vocab-gate-backend triton --vocab-gate-chunk 2048 \
    --steps 305176 --schedule-steps 305176 --microbatch 16 --seed 42 \
    --no-activation-checkpoint --no-fused-connection-grad --evaluate-test \
    --data-dir ../data --output-dir results/train --checkpoint-dir ../checkpoints \
    --formal-training --wandb-group vocab-gate-dense-10b-20261113 --run-id "$TASK_RUN_ID" \
    --resume auto --segment-seconds 1800
