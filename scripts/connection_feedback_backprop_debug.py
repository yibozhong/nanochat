"""Same-GPU paired detach ablation; bounded diagnostic, no W&B or checkpoints."""
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
from nanochat.connection_model import ConnectionConfig,build_connection_model,optimizer_for,schedule
from nanochat.packed_training import PackedData,evaluate,json_write,seed_all,train_step


def compare_one(source,detach,data,out,steps):
    seed_all(42)
    c=ConnectionConfig(streams=1,latent_feedback=True,feedback_source=source,feedback_detach=detach)
    label=source+('_detach' if detach else '_full_grad')
    model=build_connection_model(c).prepare(compile_blocks=True)
    opt,wd=optimizer_for(model)
    log_path=out/f'{label}-steps.jsonl'
    records=[];evaluations=[];peak_bytes=0;peak_reserved=0
    def dev(step):
        row=dict(step=step,tokens=step*32768)
        for passes in (1,2,4):row[f'ce_p{passes}']=evaluate(model,data,passes=passes)[0]
        assert all(torch.isfinite(torch.tensor(row[k])) for k in ('ce_p1','ce_p2','ce_p4'))
        evaluations.append(row)
        json_write(out/f'{label}-eval.json',evaluations)
        print('DEV '+json.dumps(dict(model=label,**row)),flush=True)
    dev(0)
    with log_path.open('w',buffering=1) as log:
        for step in range(steps):
            schedule(opt,step,305176,wd)
            passes=model.training_passes(42,step)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize();start=time.perf_counter()
            loss=train_step(model,opt,data,step,microbatch=16,seed=42)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            peak_bytes=max(peak_bytes,torch.cuda.max_memory_allocated())
            peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved())
            row=dict(step=step+1,passes=passes,loss=loss,seconds=elapsed)
            records.append(row);log.write(json.dumps(row)+'\n')
            if (step+1)%256==0:
                print('TRAIN '+json.dumps(dict(model=label,**row)),flush=True)
                dev(step+1)
    if evaluations[-1]['step']!=steps:dev(steps)
    # Exclude first 128 updates (compile/allocator warmup), and the first update
    # after each eval. Timing includes data, finite-gradient guard and optimizer.
    timed=[r for r in records if r['step']>128 and r['step']%256!=1]
    times={str(p):[r['seconds'] for r in timed if r['passes']==p] for p in (1,2)}
    result=dict(config=asdict(c),steps=steps,tokens=steps*32768,parameters=sum(p.numel() for p in model.parameters()),
        first_loss=records[0]['loss'],last_loss=records[-1]['loss'],
        mean_last128_train_ce=statistics.mean(r['loss'] for r in records[-128:]),
        mean_seconds_by_pass={p:statistics.mean(t) for p,t in times.items()},
        median_seconds_by_pass={p:statistics.median(t) for p,t in times.items()},
        p90_seconds_by_pass={p:sorted(t)[int(.9*(len(t)-1))] for p,t in times.items()},
        mean_step_seconds=statistics.mean(r['seconds'] for r in timed),
        training_seconds=sum(r['seconds'] for r in records),
        steady_training_seconds=sum(r['seconds'] for r in records if r['step']>128),
        peak_allocated_bytes=peak_bytes,peak_reserved_bytes=peak_reserved,
        evaluations=evaluations,cost=model_cost(c,batch=16),complete=True)
    json_write(out/f'{label}.json',result)
    return result


def check_initialization(source):
    """Check actual tensors, rather than requiring bitwise-equal CE reductions."""
    models=[]
    for detach in (True,False):
        seed_all(42)
        c=ConnectionConfig(streams=1,latent_feedback=True,feedback_source=source,feedback_detach=detach)
        models.append(build_connection_model(c).prepare())
    left,right=[m.state_dict() for m in models]
    assert left.keys()==right.keys()
    assert all(torch.equal(left[name],right[name]) for name in left)
    return dict(tensors=len(left),parameters=sum(p.numel() for p in models[0].parameters()),
        bitwise_identical=True)


def summarize_comparisons(result):
    for source in ('embedding_qkv','per_layer_qkv'):
        d=result['models'][source+'_detach'];f=result['models'][source+'_full_grad']
        assert d['parameters']==f['parameters']
        assert abs(d['first_loss']-f['first_loss'])<2e-4
        for key in ('ce_p1','ce_p2','ce_p4'):
            a,b=d['evaluations'][0][key],f['evaluations'][0][key]
            # FP32 reduction roundoff: require agreement within one ULP scale.
            assert abs(a-b)<=torch.finfo(torch.float32).eps*max(abs(a),abs(b),1.)
        result['comparisons'][source]=dict(
            p2_time_ratio=f['median_seconds_by_pass']['2']/d['median_seconds_by_pass']['2'],
            sampled_mean_time_ratio=f['mean_step_seconds']/d['mean_step_seconds'],
            peak_memory_ratio=f['peak_allocated_bytes']/d['peak_allocated_bytes'],
            peak_extra_gib=(f['peak_allocated_bytes']-d['peak_allocated_bytes'])/2**30,
            train_ce_delta=f['mean_last128_train_ce']-d['mean_last128_train_ce'],
            dev_ce_p2_delta=f['evaluations'][-1]['ce_p2']-d['evaluations'][-1]['ce_p2'],
            dev_ce_p4_delta=f['evaluations'][-1]['ce_p4']-d['evaluations'][-1]['ce_p4'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--steps',type=int,default=2048)
    a=parser.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    configure_attention('fa2');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.recompile_limit=max(32,torch._dynamo.config.recompile_limit)
    data=PackedData(a.data_dir);assert len(data.arrays['train'])>=a.steps*16
    props=torch.cuda.get_device_properties(0)
    result=dict(complete=False,formal_training=False,gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,data_manifest=data.manifest,
        seed=42,steps_per_model=a.steps,tokens_per_model=a.steps*32768,
        batch_tokens=32768,microbatch=16,schedule_steps=305176,pass_mix=[.15,.85],
        loss='final-pass CE only in both arms',models={},comparisons={},
        timing_note='Same GPU, one model resident at a time; identical init seed, data prefix, optimizer and sampled passes. First128 and post-eval steps excluded from latency. Counterbalanced arm order between architectures. No checkpointing, profiler or W&B. Peak memory is full training allocation.',
        gpu_info=dict(name=props.name,total_memory_bytes=props.total_memory,
            capability=[props.major,props.minor],multiprocessors=props.multi_processor_count))
    result['initialization_checks']={source:check_initialization(source)
        for source in ('embedding_qkv','per_layer_qkv')}
    gc.collect();torch.cuda.empty_cache()
    # Opposite order for the second pair to reduce systematic warm-cache bias.
    settings=[('embedding_qkv',True),('embedding_qkv',False),('per_layer_qkv',False),('per_layer_qkv',True)]
    for source,detach in settings:
        label=source+('_detach' if detach else '_full_grad')
        result['models'][label]=compare_one(source,detach,data,out,a.steps)
        gc.collect();torch.cuda.empty_cache();json_write(out/'summary.json',result)
    summarize_comparisons(result)
    result['complete']=True;json_write(out/'summary.json',result)
    print('FEEDBACK_BACKPROP_COMPARISON '+json.dumps(result['comparisons']),flush=True)


if __name__=='__main__':main()
