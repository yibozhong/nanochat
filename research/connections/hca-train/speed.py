"""Same-allocation production-shape throughput; diagnostics only, no W&B."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--microbatch',type=int,required=True)
    p.add_argument('--block-size',type=int,default=16)
    p.add_argument('--modes',nargs='+',choices=('full','fixed_hca','dynamic_hca'),default=['full','fixed_hca','dynamic_hca'])
    a=p.parse_args()
    import torch
    from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
    from nanochat.connection_experiment import configure_attention
    from nanochat.packed_training import PackedData, seed_all, train_step
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    configure_attention('fa2')
    data=PackedData('../data')
    report=dict(complete=False,formal_training=False,microbatch=a.microbatch,
                gpu=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                source=json.loads(Path('source.json').read_text()),modes={})
    assert 'full' in a.modes and len(set(a.modes))==len(a.modes)
    for mode in a.modes:
        seed_all(42)
        c=ConnectionConfig(variant='baseline',streams=1,attention_mode=mode,hca_backend='triton',hca_block_size=a.block_size)
        model=build_connection_model(c).prepare(compile_blocks=False)
        opt,wd=optimizer_for(model)
        seconds=[];losses=[]
        torch.cuda.reset_peak_memory_stats()
        for step in range(4):
            schedule(opt,step,305176,wd)
            torch.cuda.synchronize();start=time.perf_counter()
            losses.append(train_step(model,opt,data,step,microbatch=a.microbatch))
            torch.cuda.synchronize();seconds.append(time.perf_counter()-start)
        report['modes'][mode]=dict(config=asdict(c),seconds=seconds,losses=losses,
            steady_step_seconds=sum(seconds[1:])/3,
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        del model,opt
        torch.cuda.empty_cache()
    baseline=report['modes']['full']['steady_step_seconds']
    hca_modes=[mode for mode in a.modes if mode!='full']
    for mode in hca_modes:
        row=report['modes'][mode]
        row['relative_to_baseline']=row['steady_step_seconds']/baseline
        row['estimated_train_only_hours']=305176*row['steady_step_seconds']/3600
    report['complete']=True
    path=Path('results/hca-startup/speed.json')
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(event='HCA_TRITON_SPEED',**report),allow_nan=False),flush=True)
    # Engineering guard against repeating the diagnosed 184x/260x slowdown.
    # This is deliberately not evidence of quality or a performance promise.
    assert all(report['modes'][m]['relative_to_baseline'] < 10 for m in hca_modes), 'HCA is still >10x the same-GPU baseline; inspect before starting 10B'


if __name__=='__main__':main()
