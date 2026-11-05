"""Same-GPU old/compact8 read+aggregate forward/backward at production shape."""
import gc,json
from pathlib import Path
import torch
from nanochat.stream_mlp_connections import StreamMLPHyperConnection


def main():
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    x=torch.randn(16,2048,512,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    gy=torch.randn_like(x[...,0]);gp=torch.randn(16,2048,4,device='cuda')
    old=StreamMLPHyperConnection(512,4,0).cuda();old.reset_parameters()
    compact=StreamMLPHyperConnection(512,4,0,compact_routing=True).cuda();compact.load_state_dict(old.state_dict())
    results={}
    for name,model in [('padded_nvidia',old),('compact8',compact)]:
        def iteration():
            y,_,post=model.read(x)
            return torch.autograd.grad((y,post),(x,model.phi,model.scales,model.bias),(gy,gp))
        for _ in range(5):iteration()
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(20):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();iteration();b.record();b.synchronize();times.append(a.elapsed_time(b))
        results[name]=dict(mean_ms=sum(times)/len(times),samples_ms=times,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    root=Path('results/compact');root.mkdir(parents=True,exist_ok=True)
    (root/'micro.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results),flush=True)
if __name__=='__main__':main()
