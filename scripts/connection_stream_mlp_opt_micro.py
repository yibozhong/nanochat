"""Same-device production-size comparison, including compiled FP32 matmuls."""
import gc,json,statistics
from pathlib import Path
import torch
from nanochat.stream_mlp import stream_mlp
from nanochat.stream_mlp_optimized import optimized_stream_mlp,_dot_backward,_row_forward
from scripts.connection_stream_mlp_micro import reference


def measure(fn):
    # Capture already-compiled work: repeated replay excludes Python launch
    # starvation and allocator first-use effects. Whole-model timing is separate.
    for _ in range(8):fn()
    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):outputs=fn()
    for _ in range(5):graph.replay()
    samples=[]
    for _ in range(20):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):graph.replay()
        b.record();b.synchronize();samples.append(a.elapsed_time(b)/5)
    result=dict(mean_ms=statistics.mean(samples),median_ms=statistics.median(samples),samples_ms=samples,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    del graph,outputs;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    x=torch.randn(16,2048,512,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.randn(16,2048,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    p=torch.randn(16,2048,4,device='cuda',requires_grad=True)
    wi=(torch.randn(16,4,device='cuda')*.5).requires_grad_()
    wo=(torch.randn(4,16,device='cuda')*.25).requires_grad_()
    args=(x,y,p,wi,wo);go=torch.randn_like(x)
    functions={'legacy':stream_mlp}
    for kind in ('rowwise','hybrid'):
        functions[kind]=lambda *args,kind=kind:optimized_stream_mlp(*args,kind)
    functions['torch_compile']=torch.compile(reference,fullgraph=True)
    result=dict(gpu=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                shape=list(x.shape),precision='BF16 activation/output; FP32 mixer/intermediates; dots TF32x3',
                timing='CUDA graph replay;8 eager warmup,5 replay warmup,20 samples of5 replays',results={})
    for name,fn in functions.items():
        def fw():return fn(*args)
        def bw():
            out=fn(*args)
            return out,torch.autograd.grad(out,args,go)
        result['results'][name]=dict(forward=measure(fw),forward_backward=measure(bw))
        print(name,json.dumps(result['results'][name]),flush=True)
    # Repeat in reverse order to expose drift; select by average median.
    for name in reversed(functions):
        fn=functions[name]
        def iteration():
            out=fn(*args);return out,torch.autograd.grad(out,args,go)
        result['results'][name]['repeat']=measure(iteration)
    scores={name:statistics.mean([r['forward_backward']['median_ms'],r['repeat']['median_ms']])
        for name,r in result['results'].items() if name!='torch_compile'}
    result['scores_ms']=scores;result['winner']=min(scores,key=scores.get)
    # Inspect actual emitted instructions, rather than assuming tl.dot uses TC.
    out=torch.empty_like(x);n=x.numel()//4
    gx=torch.empty_like(x);parts=torch.empty(((n+127)//128,128),device='cuda',dtype=torch.float32)
    kernels={
        'rowwise':_row_forward[((n+255)//256,)](x,y,p,wi,wo,out,n,512,256,num_warps=4),
        'hybrid_backward':_dot_backward[((n+127)//128,)](x,wi,wo,go,gx,parts,n,128,num_warps=4)}
    result['compiled']={name:dict(registers=k.n_regs,spills=k.n_spills,
        mma_sync_instructions=k.asm['ptx'].count('mma.sync'),
        wgmma_instructions=k.asm['ptx'].count('wgmma.mma_async')) for name,k in kernels.items()}
    del out,gx,parts
    # One warmed iteration for a kernel-level CUDA time breakdown.
    result['kernel_breakdown']={}
    for name in dict.fromkeys(('legacy',result['winner'])):
        fn=functions[name]
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            z=fn(*args);grads=torch.autograd.grad(z,args,go);torch.cuda.synchronize()
        rows=[dict(name=e.key,calls=e.count,cuda_us=e.device_time_total) for e in prof.key_averages()
              if str(e.device_type).endswith('CUDA')]
        result['kernel_breakdown'][name]=sorted(rows,key=lambda r:r['cuda_us'],reverse=True)
        del z,grads
    root=Path('results/stream-mlp-opt');root.mkdir(parents=True,exist_ok=True)
    (root/'micro.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(winner=result['winner'],scores_ms=scores,compiled=result['compiled'])),flush=True)


if __name__=='__main__':main()
