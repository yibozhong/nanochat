export VOCAB_GATE_ROOT="${VOCAB_GATE_ROOT:-$PWD}"
export PATH="$VOCAB_GATE_ROOT/runtime/.venv/bin:$VOCAB_GATE_ROOT/toolchain/.venv/lib/python3.11/site-packages/nvidia/cu13/bin:/opt/project/.venv/bin:$PATH"
export CUDA_HOME="$VOCAB_GATE_ROOT/toolchain/.venv/lib/python3.11/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$VOCAB_GATE_ROOT/runtime/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
if [ -d "$VOCAB_GATE_ROOT/cuda-compat/usr/local/cuda-13.1/compat" ]; then export LD_LIBRARY_PATH="$VOCAB_GATE_ROOT/cuda-compat/usr/local/cuda-13.1/compat:$LD_LIBRARY_PATH"; fi
export PYTHONPATH="$VOCAB_GATE_ROOT/nanochat-src"
export NANOCHAT_DTYPE=bfloat16 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export NANOCHAT_BASE_DIR="$VOCAB_GATE_ROOT/cache"
export OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUBLAS_WORKSPACE_CONFIG=:4096:8
export TRITON_CACHE_DIR="$VOCAB_GATE_ROOT/triton-cache" TORCHINDUCTOR_CACHE_DIR="$VOCAB_GATE_ROOT/inductor-cache"
unset NVIDIA_TF32_OVERRIDE

export NANOCHAT_DISABLE_FA3=1
