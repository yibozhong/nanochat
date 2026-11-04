"""Bounded GPU correctness, same-device operator/full-step speed and memory."""
import argparse,json,sys,time
from pathlib import Path
import torch
from scripts.connection_moment_gate import execute
from nanochat.packed_training import json_write
from nanochat.triton_moment_attention import fused_attention

def operator_profile(modes=('none','second_moment','variance','jensen_silu')):
    from torch.profiler import profile,ProfilerActivity
    rows=[]
    for mode in modes:
        xs=[torch.randn(16,2048,4,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
        def run():
            out=fused_attention(*xs,mode)
            sum(x.float().square().mean() for x in out if x.numel()).backward()
            for x in xs:x.grad=None
        for _ in range(10):run()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(30):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();run();b.record();b.synchronize();times.append(a.elapsed_time(b))
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:run();torch.cuda.synchronize()
        names={x.key:x.count for x in prof.key_averages() if 'fwd_kernel' in x.key or 'bwd_kernel' in x.key or 'preprocess' in x.key}
        assert any('fwd_kernel' in k for k in names),names
        rows.append(dict(mode=mode,forward_backward_ms=sum(times)/len(times),samples_ms=times,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),kernels=names))
    json_write('results/operator-profile.json',rows)

def main():
    started=time.perf_counter()
    execute([sys.executable,'-m','pytest','tests/test_triton_moments.py','-v','--junitxml=results/triton-tests.xml'],'triton-tests')
    operator_profile()
    profiles=[]
    for repeat,order in enumerate([('none','variance','jensen_silu','second_moment'),('second_moment','jensen_silu','variance','none')]):
        for mode in order:
            name=f'profile-{repeat}-'+mode
            execute([sys.executable,'-m','scripts.connection_moment_gate','--profile',mode,'--kernel','fla_triton','--output','results/'+name+'.json'],name)
            profiles.append(json.loads(Path('results',name+'.json').read_text()))
    for mode in ('none','variance','jensen_silu'):
        name='reference-'+mode
        execute([sys.executable,'-m','scripts.connection_moment_gate','--profile',mode,'--output','results/'+name+'.json'],name)
        profiles.append(json.loads(Path('results',name+'.json').read_text()))
    json_write('results/triton-gate.json',dict(complete=True,source=json.loads(Path('source.json').read_text()),profiles=profiles,
        seconds=time.perf_counter()-started,formal_training=False,weights_returned=False))
if __name__=='__main__':main()
