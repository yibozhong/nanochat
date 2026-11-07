"""One actual full-size NTP update with scalar activation diagnostics.

Diagnostic overhead is never reported as throughput. Failures are recorded;
the validation orchestrator decides whether to continue to timed training.
"""
import gc
import json
import math
from pathlib import Path
import torch

import nanochat.self_gated_connections as impl
from nanochat.connection_model import ConnectionConfig,build_connection_model,optimizer_for,schedule
from nanochat.connection_experiment import configure_attention,match_vanilla_initialization
from nanochat.packed_training import seed_all,train_step
from scripts.connection_context_profile import TimingData


def stats(x):
    with torch.no_grad():
        value=x.detach().float();maximum=value.abs().max().item()
        return dict(finite=bool(torch.isfinite(value).all()),max_abs=maximum if math.isfinite(maximum) else None)


def one(variant):
    seed_all(42);model=build_connection_model(ConnectionConfig(variant=variant)).prepare(compile_blocks=True)
    match_vanilla_initialization(model,42);opt,wd=optimizer_for(model,32768);data=TimingData('../data',2048)
    events=[];current={'name':None};gate=impl.self_gate
    def observed_gate(x,weight,bias):
        events.append(dict(connection=current['name'],stage='gate_input',**stats(x)))
        out=gate(x,weight,bias);events.append(dict(connection=current['name'],stage='gate_output',**stats(out)))
        return out
    if variant=='self_gated_hc':
        impl.self_gate=observed_gate
        for name,conn in model.named_modules():
            if not isinstance(conn,impl.SelfGatedHyperConnection):continue
            original=conn.write
            def observed_write(output,residual,mixes,fn=original,name=name):
                current['name']=name
                events.append(dict(connection=name,stage='branch_output',**stats(output)))
                out=fn(output,residual,mixes)
                events.append(dict(connection=name,stage='mixed_output',**stats(out)))
                return out
            conn.write=observed_write
    result=dict(variant=variant,finite=False,optimizer_updates=0,events=events)
    try:
        schedule(opt,0,305176,wd)
        loss=train_step(model,opt,data,0,16,32768)
        result.update(finite=math.isfinite(loss),loss=loss,optimizer_updates=1)
    except Exception as exc:
        result.update(error=repr(exc),nonfinite_gradients=[n for n,p in model.named_parameters()
            if p.grad is not None and not bool(torch.isfinite(p.grad).all())])
    finally:impl.self_gate=gate
    result['first_nonfinite_event']=next((e for e in events if not e['finite']),None)
    return result


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;configure_attention('hardware_auto')
    rows={}
    for variant in ('mhc_no_sinkhorn','self_gated_hc'):
        rows[variant]=one(variant);print(json.dumps(rows[variant]),flush=True)
        gc.collect();torch.cuda.empty_cache();torch._dynamo.reset()
    value=dict(complete=True,source=json.loads(Path('source.json').read_text()),formula='x*silu(a*x+b)',
        cases=rows,passed=all(r['finite'] and r['optimizer_updates']==1 for r in rows.values()),
        no_throughput_claim=True,weights_reused_in_later_tests=False,formal_training=False)
    root=Path('results/self-gate');root.mkdir(parents=True,exist_ok=True)
    (root/'first-step-probe.json').write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
