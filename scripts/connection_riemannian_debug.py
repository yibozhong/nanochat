"""One bounded RMT stability check plus a vanilla throughput reference, no W&B."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from nanochat.connection_cost import model_cost
from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
from nanochat.packed_training import PackedData, evaluate, json_write, seed_all, train_step


@torch.no_grad()
def geometry_audit(model, x):
    from nanochat.riemannian_momentum import sphere_project, tangent_project
    h,v=model.initial_state(model.wte(x),x);rows=[]
    mode=model.config.residual_mode
    cos,sin=model.cos[:,:x.shape[1]],model.sin[:,:x.shape[1]]
    for b in model.blocks:
        for name,branch,update in (
            ('attention',lambda z:b.attention_branch(z,cos,sin),b.attn_residual),
            ('ffn',b.mlp_branch,b.mlp_residual)):
            force=branch(h.to(cos.dtype))
            next_h,next_v=update(h,v,force)
            assert torch.isfinite(next_h).all()
            row=dict(sublayer=len(rows),branch=name,hidden_rms=next_h.square().mean().sqrt().item())
            if mode != 'heavy_ball':
                row['radius_relative_error']=(next_h.square().mean(-1)-1).abs().max().item()
                assert row['radius_relative_error']<2e-6,row
            if next_v is not None:
                assert torch.isfinite(next_v).all()
                row['velocity_rms']=next_v.square().mean().sqrt().item()
                beta,gamma,eta=update.coefficients()
                u=beta*v+gamma*force.float()
                if mode=='riemannian':
                    u=tangent_project(h,u)
                    row['tangency_cosine_error']=((next_h*next_v).sum(-1).abs()
                        /(next_h.norm(dim=-1)*next_v.norm(dim=-1)).clamp_min(1e-20)).max().item()
                    assert row['tangency_cosine_error']<2e-6,row
                row['transport_relative_norm_error']=((next_v.norm(dim=-1)-u.norm(dim=-1)).abs()
                    /u.norm(dim=-1).clamp_min(1e-20)).max().item()
                assert row['transport_relative_norm_error']<2e-6,row
            rows.append(row)
            h,v=next_h,next_v
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--residual-dynamics',choices=('riemannian','heavy_ball','sphere'),default='riemannian')
    p.add_argument('--velocity-init',choices=('zero','token'),default='zero')
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    configure_attention('fa2');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    data=PackedData(a.data_dir);assert len(data.arrays['train'])>=512*16
    result=dict(complete=False,formal_training=False,gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,attention_backend='fa2',seed=42,
        stability_steps=512,microbatch=16,batch_tokens=32768,models={})
    for label,enabled,steps in [('vanilla_timing_reference',False,32),('riemannian_momentum',True,512)]:
        seed_all(42)
        c=ConnectionConfig(streams=1,residual_dynamics=a.residual_dynamics if enabled else 'vanilla',
            velocity_init=a.velocity_init if enabled else 'zero')
        m=build_connection_model(c).prepare(compile_blocks=True);opt,wd=optimizer_for(m)
        durations=[];losses=[]
        initial_dev=evaluate(m,data)[0] if enabled else None
        torch.cuda.reset_peak_memory_stats()
        with (out/(label+'-steps.jsonl')).open('w',buffering=1) as log:
            for step in range(steps):
                schedule(opt,step,305176,wd)
                torch.cuda.synchronize();begin=time.perf_counter()
                loss=train_step(m,opt,data,step,microbatch=16)
                torch.cuda.synchronize();seconds=time.perf_counter()-begin
                row=dict(model=label,step=step+1,loss=loss,seconds=seconds)
                log.write(json.dumps(row)+'\n');losses.append(loss)
                if 16<=step<32:durations.append(seconds)
                if step<2 or (step+1)%64==0:print(json.dumps(row),flush=True)
        info=dict(config=asdict(c),parameters=sum(p.numel() for p in m.parameters()),
            steps=steps,mean_step_seconds=statistics.mean(durations),median_step_seconds=statistics.median(durations),
            tokens_per_second=32768/statistics.mean(durations),first_loss=losses[0],last_loss=losses[-1],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            theoretical_cost=model_cost(c,batch=16))
        if enabled:
            info.update(initial_dev_ce=initial_dev,final_dev_ce=evaluate(m,data)[0],coefficients=m.residual_metrics())
            x,y=data.batch('dev',0,16)
            info['geometry']=geometry_audit(m,x)
            assert statistics.mean(losses[-32:])<statistics.mean(losses[:32]) and max(losses)<20
            assert info['final_dev_ce']<initial_dev
            assert info['peak_allocated_bytes']<.9*torch.cuda.get_device_properties(0).total_memory
            del x,y
        # Measure supported operator FLOPs on the actual compiled training path.
        # FA2, CCE, fused elementwise and optimizer kernels are not fully counted
        # by PyTorch; never label this partial sum as total hardware FLOPs.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],with_flops=True) as prof:
            train_step(m,opt,data,0,microbatch=16)
            torch.cuda.synchronize()
        counted=[dict(operator=e.key,flops=e.flops,count=e.count) for e in prof.key_averages() if e.flops]
        info['profiler_supported_flops']=sum(e['flops'] for e in counted)
        info['profiler_counted_operators']=counted
        info['profiler_coverage']='Partial operator count on compiled fwd/backward/optimizer. FA2/CCE/fused kernels not fully covered; not total hardware FLOPs.'
        result['models'][label]=info
        json_write(out/'summary.json',result)
        del m,opt,prof;torch._dynamo.reset();gc.collect();torch.cuda.empty_cache()
    base=result['models']['vanilla_timing_reference'];rmt=result['models']['riemannian_momentum']
    result['time_ratio']=rmt['median_step_seconds']/base['median_step_seconds']
    result['additional_peak_allocated_bytes']=rmt['peak_allocated_bytes']-base['peak_allocated_bytes']
    # Predeclared engineering guard; a failure stops the formal run for review.
    assert result['time_ratio']<1.75,result['time_ratio']
    result['complete']=True;json_write(out/'summary.json',result)
    print(json.dumps(dict(profile_complete=True,time_ratio=result['time_ratio'],
        rmt_step_seconds=rmt['median_step_seconds'],dev_ce=rmt['final_dev_ce'])),flush=True)


if __name__=='__main__':main()
