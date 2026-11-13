"""Bounded, same-allocation benchmarks. No W&B or formal training."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import torch
import triton

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.ffn_readout import initialize_gradmax, readout_reference
from nanochat.ffn_readout_triton import fused_readout, _forward, _backward
from nanochat.packed_training import seed_all, PackedData, train_step, evaluate


def run_model(args):
    seed_all(42);configure_attention('fa2');torch.backends.cuda.matmul.allow_tf32=False
    c=ConnectionConfig(streams=1,ffn_readout_width=args.width,ffn_readout_backend=args.backend)
    model=ConnectionModel(c).prepare(compile_blocks=True)
    data=PackedData('../data')
    init=None
    if args.width:
        x,y=data.batch('train',0,1)
        torch.cuda.synchronize();calibration_start=time.perf_counter()
        init=initialize_gradmax(model,x,y)
        torch.cuda.synchronize();init['seconds']=time.perf_counter()-calibration_start
    opt,wd=optimizer_for(model)
    # Warm all compilation/autotuning and optimizer state before timing.
    for step in range(10):
        schedule(opt,step,305176,wd);train_step(model,opt,data,step)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    times=[];losses=[]
    for step in range(10,10+args.steps):
        schedule(opt,step,305176,wd)
        torch.cuda.synchronize();start=time.perf_counter()
        loss=train_step(model,opt,data,step)
        torch.cuda.synchronize();times.append(1000*(time.perf_counter()-start));losses.append(loss)
    memory=torch.cuda.max_memory_allocated()/2**30
    gradients={}
    if args.width:
        gradients=dict(branch_down=model.blocks[-1].mlp.c_proj.weight.grad[c.width:].norm().item(),
            branch_head=model.lm_head.weight.grad[:,c.width:].norm().item(),
            last_ffn_up=model.blocks[-1].mlp.c_fc.weight.grad.norm().item())
        assert all(v>0 for v in gradients.values())
    dev,tokens=evaluate(model,data,max_rows=64)
    report=dict(formal_training=False,source=json.loads(Path('source.json').read_text()),
        data_manifest=data.manifest,attention_backend='fa2',gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        triton=triton.__version__,config=asdict(c),parameters=sum(p.numel() for p in model.parameters()),
        init=init,warmup_updates=10,measured_updates=args.steps,median_ms=statistics.median(times),
        p10_ms=sorted(times)[len(times)//10],p90_ms=sorted(times)[9*len(times)//10],
        peak_allocated_gib=memory,first_loss=losses[0],last_loss=losses[-1],
        all_updates_finite=True,gradients=gradients,dev_ce=dev,dev_tokens=tokens)
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            train_step(model,opt,data,10+args.steps)
        Path(args.output+'.profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
    return report


def run_tiles(args):
    seed_all(42)
    m,d,r=32768,512,64
    x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    p=torch.randn(m,d+r,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    g=torch.randn_like(p);y=torch.empty_like(p);inv=torch.empty(m,device='cuda')
    dx=torch.empty_like(x);dp=torch.empty_like(p)
    rows=[]
    for bm in (1,2,4,8,16):
        for warps in (4,8):
            kw=dict(num_warps=warps,enable_fp_fusion=False)
            f=lambda:_forward[(triton.cdiv(m,bm),)](x,p,y,inv,m,d,r,bm,d,r,**kw)
            b=lambda:_backward[(triton.cdiv(m,bm),)](x,p,g,inv,dx,dp,m,d,r,bm,d,r,**kw)
            f();b();torch.cuda.synchronize()
            ref=readout_reference(x,p);ref_dx,ref_dp=torch.autograd.grad(ref,(x,p),g)
            errors=[((a.float()-v.float()).norm()/v.float().norm()).item() for a,v in [(y,ref),(dx,ref_dx),(dp,ref_dp)]]
            assert max(errors)<.005,(bm,warps,errors)
            rows.append(dict(bm=bm,warps=warps,forward_ms=triton.testing.do_bench(f),backward_ms=triton.testing.do_bench(b),relative_errors=errors))
    def pair(fn):
        out=fn(x,p)
        torch.autograd.grad(out,(x,p),g)
    reference=torch.compile(readout_reference,fullgraph=True)
    pair(reference)
    return dict(formal_training=False,gpu=torch.cuda.get_device_name(),shape=[m,d,r],tiles=rows,
        triton_pair_ms=triton.testing.do_bench(lambda:pair(fused_readout)),
        triton_separate_gradient_pair_ms=triton.testing.do_bench(lambda:pair(lambda x,p:fused_readout(x,p,False))),
        compiled_reference_pair_ms=triton.testing.do_bench(lambda:pair(reference)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--width',type=int,default=64)
    parser.add_argument('--backend',choices=['triton','reference'],default='triton')
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--tiles',action='store_true')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    report=run_tiles(args) if args.tiles else run_model(args)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
