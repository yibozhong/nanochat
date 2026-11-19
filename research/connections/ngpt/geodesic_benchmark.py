"""Compare the deployed geodesic write to a fused Triton write on the same GPU."""
import gc
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch
import triton

from nanochat import lookahead
from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
from nanochat.geodesic_residual import geodesic_split_write
from nanochat.momentum import normalize_residual, tangent_project
from nanochat.packed_training import PackedData, json_write, seed_all, train_step
from research.connections.ngpt.geodesic_triton import fused_geodesic_write

TOKENS = 32768
WARM, MEASURE = 16,32
REPORT = Path('results/comparison.json')


def pytest_configure(config):
    # Only the explicitly enabled benchmark plugin changes the train/inference path.
    lookahead.geodesic_split_write = fused_geodesic_write


def config(**kwargs):
    return ConnectionConfig(streams=1,residual_dynamics='ngpt_lookahead',ngpt_velocity_mode='lookahead_geodesic_split',
                            velocity_init='token_position',residual_projection_init='normal',ngpt_force_rms_floor=1e-6,**kwargs)


def inputs(n,d,case,dtype):
    h=normalize_residual(torch.randn(n,d,device='cuda'))
    v=tangent_project(h,torch.randn_like(h))*.13
    f=torch.randn_like(h).to(dtype)
    if case in ('aligned','antipodal','near_antipodal'):
        h.fill_(1.)
        v=tangent_project(h,v)
        f=h.clone() if case=='aligned' else -h.clone()
        if case=='near_antipodal':
            f+=1e-3*torch.randn_like(f)
        f=f.to(dtype)
    if case=='force_floor':
        f[:n//2].zero_()
        f[n//2:]*=1e-7
    if case=='zero_velocity':
        v.zero_()
    steps=(.00001,.000023) if case=='tiny_step' else (.14,.31)
    return tuple(x.requires_grad_() for x in (h,v,f,*[h.new_tensor(a) for a in steps]))


def gradient_checks():
    result={}
    for width,case,dtype in [(128,'ordinary',torch.float32),(512,'ordinary',torch.bfloat16),
                             (512,'force_floor',torch.float32),(512,'zero_velocity',torch.float32),
                             (512,'aligned',torch.float32),(512,'antipodal',torch.float32),
                             (512,'near_antipodal',torch.float32),(512,'tiny_step',torch.float32)]:
        seed_all()
        xs=inputs(35,width,case,dtype)
        ref=geodesic_split_write(*xs,.5,1e-6)
        actual=fused_geodesic_write(*xs,.5,1e-6)
        upstream=tuple(torch.randn_like(x) for x in ref)
        for x,y in zip(actual,ref):
            torch.testing.assert_close(x,y,atol=5e-5,rtol=5e-5)
        expected=torch.autograd.grad(ref,xs,upstream)
        computed=torch.autograd.grad(actual,xs,upstream)
        errors={}
        for key,x,y in zip(('h','v','force','perp','parallel'),computed,expected):
            assert torch.isfinite(x).all() and torch.isfinite(y).all()
            diff=(x.double()-y.double()).norm().item()
            norm=y.double().norm().item()
            errors[key]=diff/max(norm,1e-10)
            assert diff <= (0.006 if x.dtype==torch.bfloat16 else 0.001)*norm+2e-5,(case,key,diff,norm)
        result[f'{width}_{case}']=errors
    return result


def model_gradient_check():
    """Compare the complete compiled NTP backward with an active velocity read."""
    seed_all()
    c=config(depth=2,width=128,sequence_length=32,heads=1,vocab_size=256,lookahead_mu_init=.3)
    ids=torch.randint(0,256,(2,17),device='cuda')
    results=[]
    for fn in (geodesic_split_write,fused_geodesic_write):
        lookahead.geodesic_split_write=fn
        seed_all()
        model=build_connection_model(c).prepare(compile_blocks=True)
        with torch.no_grad():
            for block in model.blocks:
                block.attn_residual.raw_parallel.add_(.7)
                block.mlp_residual.raw_parallel.sub_(.4)
        loss=model(ids[:,:-1].contiguous(),ids[:,1:].contiguous())
        loss.backward()
        results.append((loss.item(),{n:p.grad.detach().float().cpu() for n,p in model.named_parameters()}))
    (first,ref),(second,actual)=results
    assert abs(first-second)<.001
    errors={}
    for name,x in actual.items():
        y=ref[name]
        diff=(x-y).norm().item()
        norm=y.norm().item()
        assert torch.isfinite(x).all() and diff <= .04*norm+2e-7,(name,diff,norm)
        errors[name]=diff/max(norm,1e-10)
    return dict(loss=[first,second],relative_gradient_errors=errors)


def time_operator(fn,xs):
    fn=torch.compile(fn,fullgraph=True,dynamic=False,options={'emulate_precision_casts':True})
    upstream=None
    samples=[]
    for i in range(30):
        begin,mid,end=(torch.cuda.Event(enable_timing=True) for _ in range(3))
        begin.record()
        out=fn(*xs)
        if not isinstance(out,tuple):
            out=(out,)
        mid.record()
        if upstream is None:
            upstream=tuple(torch.randn_like(x) for x in out)
        torch.autograd.grad(out,xs,upstream)
        end.record()
        end.synchronize()
        if i>=10:
            samples.append((begin.elapsed_time(mid),mid.elapsed_time(end)))
    return dict(forward_ms=statistics.median(x for x,_ in samples),backward_ms=statistics.median(y for _,y in samples))


def operator_timings():
    seed_all()
    xs=inputs(TOKENS,512,'ordinary',torch.bfloat16)
    h,v=xs[:2]
    mu=h.new_tensor([.3],requires_grad=True)
    def query(h,v,mu):
        offset=mu[0]*v
        return (h+offset)*torch.rsqrt(1+offset.square().mean(-1,keepdim=True))
    out={'unchanged_query':time_operator(query,(h,v,mu))}
    for key,fn in [('inductor',geodesic_split_write),('triton',fused_geodesic_write)]:
        def write(h,v,f,a,b):
            return fn(h,v,f,a,b,.5,1e-6)
        out[key]=time_operator(write,xs)
    return out


def benchmark(mode,data):
    lookahead.geodesic_split_write=fused_geodesic_write if mode=='triton' else geodesic_split_write
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    seed_all()
    model=build_connection_model(config()).prepare(compile_blocks=True)
    opt,wd=optimizer_for(model)
    durations=[]
    losses=[]
    for i in range(WARM+MEASURE):
        if i==WARM: torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start=time.perf_counter()
        schedule(opt,i,305176,wd)
        losses.append(train_step(model,opt,data,i,microbatch=16,batch_tokens=TOKENS))
        torch.cuda.synchronize()
        if i>=WARM:
            durations.append(1000*(time.perf_counter()-start))
    assert all(math.isfinite(x) for x in losses) and losses[-1]<losses[0]
    return dict(median_ms=statistics.median(durations),step_ms=durations,losses=losses,
                peak_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters()))


def main():
    props=torch.cuda.get_device_properties(0)
    assert torch.cuda.device_count()==1 and props.major==9 and ('H100' in props.name or 'H200' in props.name)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    configure_attention('fa2')
    r=dict(complete=False,formal_training=False,gpu=props.name,gpu_uuid=str(props.uuid),torch=torch.__version__,
           triton=triton.__version__,source=json.loads(Path('source.json').read_text()),rounds={},
           warmup=WARM,measured=MEASURE,batch_tokens=TOKENS)
    json_write(REPORT,r)
    try:
        r['operator_checks']=gradient_checks()
        print('VJP_CHECKS_PASSED',flush=True)
        r['model_gradient_check']=model_gradient_check()
        json_write(REPORT,r)
        subprocess.run([sys.executable,'-m','pytest','-q','tests/test_geodesic_write.py::test_active_write_cache_and_compiled_training',
                        'tests/test_geodesic_write.py::test_exact_optimizer_checkpoint_next_update',
                        '-p','research.connections.ngpt.geodesic_benchmark','--junitxml=results/fused-model-tests.xml'],check=True)
        r['write_only_gpu_ms']=operator_timings()
        print('WRITE_TIMINGS',r['write_only_gpu_ms'],flush=True)
        json_write(REPORT,r)
        data=PackedData('../data')
        for mode in ('inductor','triton','triton','inductor'):
            row=benchmark(mode,data)
            r['rounds'].setdefault(mode,[]).append(row)
            print('STEP_MEDIAN_MS',mode,row['median_ms'],flush=True)
            json_write(REPORT,r)
        r['comparison']={}
        for mode,rows in r['rounds'].items():
            ms=statistics.mean(x['median_ms'] for x in rows)
            r['comparison'][mode]=dict(ms=ms,tokens_per_second=TOKENS*1000/ms,
                                       peak_bytes=max(x['peak_bytes'] for x in rows),
                                       round_ms=[x['median_ms'] for x in rows])
        ref,new=r['comparison']['inductor']['ms'],r['comparison']['triton']['ms']
        r['step_time_reduction_pct']=100*(1-new/ref)
        r['complete']=True
        print('FINAL_COMPARISON',r['comparison'],flush=True)
    except Exception:
        r['error']=traceback.format_exc()
        raise
    finally:
        json_write(REPORT,r)


if __name__=='__main__':
    main()
