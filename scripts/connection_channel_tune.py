"""Bounded launch-geometry benchmark of the already-correct local kernels."""
import gc,json
from pathlib import Path
import torch
import triton as tr
from nanochat.channel_mixing import _forward,_backward_mix,_backward_branch


def bench(fn):
    for _ in range(3):fn()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(10):fn()
    b.record();b.synchronize()
    return a.elapsed_time(b)/10


def main():
    torch.manual_seed(42);n=32768;c=512
    x=torch.randn(16,2048,c,4,device='cuda',dtype=torch.bfloat16)
    y=torch.randn(16,2048,c,device='cuda',dtype=torch.bfloat16)
    post=torch.randn(16,2048,4,device='cuda');w=torch.randn(c,4,4,4,device='cuda')*.5
    bias=torch.eye(4,device='cuda').expand(c,4,4).contiguous();a=torch.tensor([.01],device='cuda')
    out=torch.empty_like(x);go=torch.randn_like(x);gx=torch.empty_like(x);gy=torch.empty_like(y);gp=torch.empty_like(post)
    root=Path('results/channel-tune');root.mkdir(parents=True,exist_ok=True)
    results=dict(gpu=torch.cuda.get_device_name(),forward=[],backward=[],branch=[])
    def save():
        (root/'timings.json').write_text(json.dumps(results,indent=2)+'\n')
    for bt,bc,warps in [(32,8,4),(32,8,8),(64,8,4),(64,8,8),(128,8,8),(32,32,8),(16,32,4),(16,64,8),(32,16,4),(32,16,8)]:
        try:
            fn=lambda:_forward[(tr.cdiv(n,bt),tr.cdiv(c,bc))](x,y,post,w,bias,a,out,n,c,bt,bc,num_warps=warps)
            ms=bench(fn);row=dict(bt=bt,bc=bc,warps=warps,ms=ms)
        except Exception as e:row=dict(bt=bt,bc=bc,warps=warps,error=str(e)[:1000])
        results['forward'].append(row);save();print('FORWARD',json.dumps(row),flush=True)
    for bt,bc,warps in [(128,4,4),(128,4,8),(128,4,16),(128,8,8),(128,8,16),(64,8,4),(64,8,8),(64,16,8),(32,16,4),(32,32,8),(64,16,16),(256,4,8),(256,4,16)]:
        parts=torch.empty((tr.cdiv(n,bt),c,81),device='cuda')
        try:
            fn=lambda:_backward_mix[(tr.cdiv(n,bt),tr.cdiv(c,bc))](x,w,bias,a,go,gx,parts,n,c,bt,bc,num_warps=warps)
            ms=bench(fn)
            reduce_ms=bench(lambda:parts.sum(0))
            row=dict(bt=bt,bc=bc,warps=warps,ms=ms,reduce_ms=reduce_ms,total_ms=ms+reduce_ms,workspace_bytes=parts.numel()*4)
        except Exception as e:row=dict(bt=bt,bc=bc,warps=warps,error=str(e)[:1000])
        results['backward'].append(row);save();print('BACKWARD',json.dumps(row),flush=True)
        del parts;gc.collect();torch.cuda.empty_cache()
    for warps in [4,8,16]:
        fn=lambda:_backward_branch[(n,)](y,post,go,gy,gp,n,c,tr.next_power_of_2(c),num_warps=warps)
        row=dict(warps=warps,ms=bench(fn));results['branch'].append(row);save();print('BRANCH',json.dumps(row),flush=True)
    results['complete']=True;results['source']=json.loads(Path('source.json').read_text());save()
if __name__=='__main__':main()
