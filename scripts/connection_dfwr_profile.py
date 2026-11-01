"""Bounded DFWR end-to-end GPU measurements, never a W&B run."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch

from nanochat.connection_model import ConnectionConfig,build_connection_model,optimizer_for,schedule
from nanochat.connection_experiment import configure_attention
from nanochat.depth_fast_weights import state_memory_estimate
from nanochat.packed_training import PackedData,seed_all,train_step


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--microbatch',type=int,default=1)
    p.add_argument('--bottleneck',type=int,default=32)
    p.add_argument('--steps',type=int,default=12)
    p.add_argument('--variant',choices=('baseline','dfwr'),default='dfwr')
    p.add_argument('--output',required=True)
    a=p.parse_args();seed_all(42);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;configure_attention('sdpa_cudnn')
    c=ConnectionConfig(variant=a.variant,fw_bottleneck=a.bottleneck,activation_checkpoint=a.variant=='dfwr')
    model=build_connection_model(c).prepare(compile_blocks=True)
    opt,wd=optimizer_for(model);data=PackedData('../data')
    durations=[];losses=[];torch.cuda.reset_peak_memory_stats()
    for step in range(a.steps):
        schedule(opt,step,305176,wd)
        torch.cuda.synchronize();start=time.perf_counter()
        loss=train_step(model,opt,data,step,microbatch=a.microbatch)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        losses.append(loss)
        if step>=2:durations.append(elapsed)
        print(json.dumps(dict(step=step+1,loss=loss,seconds=elapsed)),flush=True)
    record=dict(complete=True,variant=a.variant,microbatch=a.microbatch,bottleneck=a.bottleneck,
        gpu=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in model.parameters()),
        logging_backend='local',warmup=2,timed_steps=len(durations),losses=losses,
        mean_seconds=statistics.mean(durations),step_seconds=durations,
        tokens_per_second=32768/statistics.mean(durations),
        estimated_10b_training_hours=305176*statistics.mean(durations)/3600,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        state_estimate=state_memory_estimate(a.microbatch,2048,512,a.bottleneck,8))
    Path(a.output).write_text(json.dumps(record,indent=2)+'\n')
    assert all(math.isfinite(x) for x in losses)
    assert record['peak_reserved_bytes']<.9*22000*1024**2,record


if __name__=='__main__':main()
