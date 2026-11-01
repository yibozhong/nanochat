"""Bounded same-GPU full-step measurements; local JSON only."""
import argparse
import json
import statistics
import time
from pathlib import Path
import torch
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.packed_training import seed_all, PackedData, train_step


def main():
    p=argparse.ArgumentParser();p.add_argument('--lm-head-transform',required=True,choices=('linear','nonlinear_lmhead','nonlinear_residual_lmhead'));p.add_argument('--output',required=True)
    p.add_argument('--schedule-steps', type=int, default=305176)
    a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    configure_attention('sdpa_cudnn');seed_all(42)
    m=ConnectionModel(ConnectionConfig(variant='baseline',lm_head_transform=a.lm_head_transform)).prepare(compile_blocks=True)
    match_vanilla_initialization(m,42);opt,wd=optimizer_for(m);data=PackedData('../data')
    durations=[]
    torch.cuda.reset_peak_memory_stats()
    for step in range(50):
        schedule(opt,step,a.schedule_steps,wd)
        torch.cuda.synchronize();start=time.perf_counter()
        loss=train_step(m,opt,data,step)
        torch.cuda.synchronize();dt=time.perf_counter()-start
        if step>=20:durations.append(dt)
    r=dict(lm_head_transform=a.lm_head_transform,gpu=torch.cuda.get_device_name(),warmup=20,timed_steps=30,
           mean_seconds=statistics.mean(durations),median_seconds=statistics.median(durations),
           step_seconds=durations,final_loss=loss,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
           peak_allocated_bytes=torch.cuda.max_memory_allocated(),logging_backend='local')
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
    assert r['peak_reserved_bytes'] < .9*22000*1024**2


if __name__=='__main__':main()
