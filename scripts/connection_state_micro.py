"""Same-shape local read/write forward/backward workspace audit; no optimizer."""
import gc
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from nanochat.state_tokens import SlotAttention


def measure(nq, nk, backend):
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    m = SlotAttention(512, 4, backend).cuda()
    q = torch.randn(1, 32768, nq, 512, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(1, 32768, nk, 512, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    probe = torch.randn_like(q)
    fn = torch.compile(m, fullgraph=True, dynamic=False)
    def step():
        m.zero_grad(set_to_none=True); q.grad = None; kv.grad = None
        y = fn(q, kv); y.backward(probe)
    times = []
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for _ in range(3): step()
        m.zero_grad(set_to_none=True); q.grad = None; kv.grad = None
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        initial = torch.cuda.memory_allocated()
        for _ in range(6):
            torch.cuda.synchronize(); begin = time.perf_counter()
            step(); torch.cuda.synchronize(); times.append(time.perf_counter()-begin)
        result = dict(nq=nq, nk=nk, backend=backend, folded_batch=32768, heads=4, head_dimension=128,
            median_seconds=statistics.median(times), step_seconds=times,
            initial_allocated_bytes=initial, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        assert all(torch.isfinite(p.grad).all() for p in m.parameters())
        assert torch.isfinite(q.grad).all() and torch.isfinite(kv.grad).all()
        if backend == 'sdpa':
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as trace:
                step(); torch.cuda.synchronize()
            allocations = [dict(name=e.name, device_memory_bytes=e.device_memory_usage)
                           for e in trace.events() if e.device_memory_usage >= 1024**3]
            result['large_device_allocations'] = sorted(allocations,
                key=lambda x:x['device_memory_bytes'], reverse=True)[:12]
    del fn, m, q, kv, probe
    torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for nq, nk in ((1,4), (4,5)):
        for backend in ('sdpa','auto'):
            result = measure(nq, nk, backend)
            rows.append(result); print(json.dumps(result), flush=True)
    Path('results/state-slot-micro.json').write_text(json.dumps(dict(
        gpu=torch.cuda.get_device_name(), checks='Fixed weights;forward/backward only;no optimizer/W&B',
        theoretical_flash_dq_workspace_bytes=32768*128*4*128*4, results=rows), indent=2)+'\n')


if __name__ == '__main__': main()
