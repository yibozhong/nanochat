"""Production-shape GPU forward/backward timings, independent of model training."""
import json
from pathlib import Path
import torch
from nanochat.dilated_attention import sparse_attention
from nanochat.dilated_pattern import dilation_schedule
from nanochat.fla_attention import fused_attention
from nanochat.flash_attention import flash_attn


def main():
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    xs=[torch.randn(16,2048,4,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    go=torch.randn_like(xs[0]);results={}
    cases=[('fa3_full',lambda q,k,v:flash_attn.flash_attn_func(q,k,v,causal=True)),('fla_full',fused_attention)]
    cases += [(f'stride_{s}',lambda q,k,v,s=s:sparse_attention(q,k,v,s,128,124,4)) for s in dilation_schedule(2048,8)[:-1]]
    for name,fn in cases:
        fn=torch.compile(fn,fullgraph=True,dynamic=False)
        def run():torch.autograd.grad(fn(*xs),xs,go)
        for _ in range(8):run()
        torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):run()
        end.record();end.synchronize()
        record=dict(forward_backward_ms=start.elapsed_time(end)/20)
        if name in ('fa3_full','fla_full','stride_1','stride_14'):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                run();torch.cuda.synchronize()
            record['device_events']=[dict(name=e.name,microseconds=e.device_time_total) for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
        results[name]=record;print(name,json.dumps(record),flush=True)
        torch._dynamo.reset()
    data=dict(gpu=torch.cuda.get_device_name(),shape=list(xs[0].shape),results=results,
        measurement='CUDA events over20forward+VJP calls after8warmups, fixed weights/no optimizer; full-step timing separately.')
    Path('results/dilated/micro.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
