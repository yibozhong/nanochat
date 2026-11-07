"""One production NTP update for original HC and both sigmoid implementations."""
import argparse
import gc
import json
import math
from pathlib import Path
import torch
import nanochat.stream_gated_connections as connections
from nanochat.stream_gate_mixing import stream_gate_mix,compiled_stream_gate_reference
from nanochat.connection_model import ConnectionConfig,build_connection_model,optimizer_for,schedule
from nanochat.connection_experiment import configure_attention,match_vanilla_initialization
from nanochat.packed_training import seed_all,train_step
from scripts.connection_context_profile import TimingData


def stats(x):
    with torch.no_grad():
        maximum=x.detach().float().abs().max().item()
        return dict(finite=bool(torch.isfinite(x).all()),max_abs=maximum if math.isfinite(maximum) else None)


def one(backend,micro):
    variant='mhc_no_sinkhorn' if backend=='hc' else 'stream_gated_hc'
    seed_all(42);model=build_connection_model(ConnectionConfig(variant=variant)).prepare(compile_blocks=True)
    match_vanilla_initialization(model,42);opt,wd=optimizer_for(model,32768);data=TimingData('../data',2048)
    events=[];fn=compiled_stream_gate_reference if backend=='torch' else stream_gate_mix
    def observed(*args):
        events.append(dict(stage='gate_input',**stats(args[0])))
        out=fn(*args);events.append(dict(stage='mixed_output',**stats(out)))
        return out
    connections.stream_gate_mix=observed
    result=dict(backend=backend,variant=variant,finite=False,optimizer_updates=0,events=events)
    try:
        schedule(opt,0,305176,wd);loss=train_step(model,opt,data,0,micro,32768)
        result.update(finite=math.isfinite(loss),loss=loss,optimizer_updates=1)
    except Exception as exc:
        result.update(error=repr(exc),nonfinite_gradients=[n for n,p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())])
    finally:connections.stream_gate_mix=stream_gate_mix
    result['first_nonfinite_event']=next((e for e in events if not e['finite']),None)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--microbatch',type=int,default=16);a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;configure_attention('hardware_auto')
    rows={}
    for backend in ('hc','fused','torch'):
        rows[backend]=one(backend,a.microbatch);print(json.dumps(rows[backend]),flush=True)
        gc.collect();torch.cuda.empty_cache();torch._dynamo.reset()
    value=dict(complete=True,source=json.loads(Path('source.json').read_text()),cases=rows,
        passed=all(r['finite'] and r['optimizer_updates']==1 for r in rows.values()),
        no_throughput_claim=True,formal_training=False)
    root=Path('results/stream-gate');root.mkdir(parents=True,exist_ok=True)
    (root/'first-step-probe.json').write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
