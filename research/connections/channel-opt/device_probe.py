"""Actually allocate/execute/backpropagate before expensive validation or training."""
import ctypes
import json
import os
from pathlib import Path
import subprocess

record = dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
              hostname=os.uname().nodename, pid=os.getpid(), passed=False)
out = Path('results/device-probe.json')
try:
    proc = subprocess.run(['nvidia-smi',
        '--query-gpu=index,uuid,name,driver_version,compute_mode', '--format=csv,noheader'],
        text=True, capture_output=True, timeout=15)
    record['nvidia_smi'] = dict(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
except (OSError, subprocess.TimeoutExpired) as exc:
    record['nvidia_smi_unavailable'] = str(exc)
try:
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    record.update(gpu=props.name, uuid=str(props.uuid), capability=[props.major, props.minor],
                  torch=torch.__version__, runtime=torch.version.cuda)
    version = ctypes.c_int()
    assert ctypes.CDLL('libcuda.so.1').cuDriverGetVersion(ctypes.byref(version)) == 0
    record['loaded_cuda_driver_version'] = version.value
    record['libcuda_mappings'] = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                        if 'libcuda.so' in line})
    x = torch.ones((256, 256), device='cuda', dtype=torch.bfloat16, requires_grad=True)
    y = (x @ x).float().mean()
    y.backward()
    torch.cuda.synchronize()
    assert y.item() == 256 and torch.isfinite(x.grad).all()
    record.update(passed=True, allocation_and_matmul_backward=True)
except BaseException as exc:
    record['error'] = repr(exc)
    raise
finally:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)
