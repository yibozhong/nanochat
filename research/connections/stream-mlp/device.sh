# Sourced after the CUDA runtime/compatibility environment is ready.
TASK_ATTENTION_BACKEND=$(python - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count()==1
major,minor=torch.cuda.get_device_capability()
if major==9:
    print('fa3')
elif (major,minor)==(8,0) and 'A100' in torch.cuda.get_device_name():
    print('fa2')
else:
    raise RuntimeError('Expected Hopper or A100')
PY
)
export STREAM_MLP_ATTENTION_BACKEND="$TASK_ATTENTION_BACKEND"
if [ "$TASK_ATTENTION_BACKEND" = fa3 ]; then
    unset NANOCHAT_DISABLE_FA3
    export NANOCHAT_FA3_REVISION=de87b9b5af06dd9984df595bef90b2eba44b181a
    export HF_HOME="$TASK_ROOT/fa3-cache"
    HF_HUB_OFFLINE=0 python - <<'PY'
import os
from kernels import get_kernel
module=get_kernel('varunneal/flash-attention-3',revision=os.environ['NANOCHAT_FA3_REVISION'])
assert hasattr(module.flash_attn_interface,'flash_attn_func')
print('Hopper: pinned FA3; NVIDIA mHC uses its TMA projection',flush=True)
PY
else
    export NANOCHAT_DISABLE_FA3=1
    echo 'A100: embedded FA2; NVIDIA mHC selects its non-TMA projection'
fi
