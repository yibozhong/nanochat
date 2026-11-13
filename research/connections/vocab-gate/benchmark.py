"""Bounded GPU timing and 200-update finite training; no W&B or retained weights."""
import argparse,gc,json,statistics,time
from pathlib import Path
from dataclasses import asdict
import torch
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for,schedule
from nanochat.connection_experiment import configure_attention
from nanochat.packed_training import PackedData,seed_all,train_step,evaluate,json_write
from nanochat.vocab_gate_triton import linear_gate_cross_entropy


def head_timing():
    seed_all(42)
    h=torch.randn(32768,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    w=(torch.randn(65536,512,device='cuda')*.01).requires_grad_()
    y=torch.randint(32768,(32768,),device='cuda')
    rows=[]
    for kernel in ('row','tiled'):
        for chunk in (512,1024,2048,4096,8192):
            durations=[]
            for i in range(13):
                h.grad=None;w.grad=None
                torch.cuda.synchronize();start=time.perf_counter()
                loss=linear_gate_cross_entropy(h,w,y,chunk_size=chunk,kernel=kernel)
                loss.backward();torch.cuda.synchronize()
                if i>=3:durations.append((time.perf_counter()-start)*1000)
            row=dict(kernel=kernel,chunk=chunk,median_ms=statistics.median(durations),loss=loss.item())
            rows.append(row);print(json.dumps(row),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='../data');p.add_argument('--output',default='results/benchmark.json');a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;configure_attention('fa2')
    tuning=head_timing();gc.collect();torch.cuda.empty_cache()
    data=PackedData(a.data_dir)
    models=[]
    for gate in (False,True):
        seed_all(42);m=ConnectionModel(ConnectionConfig(streams=1,vocab_gate=gate)).prepare(compile_blocks=True)
        o,w=optimizer_for(m);models.append((m,o,w))
    timings=[[],[]]
    # Interleave updates on the same GPU and data, after separate compilation.
    for i in range(50):
        for k,(m,o,w) in enumerate(models):
            schedule(o,i,305176,w)
            torch.cuda.synchronize();start=time.perf_counter()
            loss=train_step(m,o,data,i,microbatch=16)
            torch.cuda.synchronize()
            if i>=10:timings[k].append((time.perf_counter()-start)*1000)
    base_ms,gate_ms=map(statistics.median,timings)
    # The baseline diagnostics end here. Continue only the candidate to 200.
    m,o,w=models[1];models.clear();gc.collect();torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for i in range(50,200):
        schedule(o,i,305176,w);loss=train_step(m,o,data,i,microbatch=16)
    assert m.lm_head.weight.grad[32768:].norm()>0
    assert m.lm_head.weight[32768:].norm()>0
    dev,_=evaluate(m,data,microbatch=16,max_rows=64)
    result=dict(source=json.loads(Path('source.json').read_text()),config=asdict(m.config),
        gpu=torch.cuda.get_device_name(),head_tuning=tuning,baseline_step_ms=base_ms,gate_step_ms=gate_ms,
        overhead_percent=100*(gate_ms/base_ms-1),finite_training_steps=200,final_train_ce=loss,
        diagnostic_dev_ce=dev,gate_gradient_nonzero=True,trained_gate_nonzero=True,
        gate_weight_norm=m.lm_head.weight[32768:].norm().item(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    # Test the actual trained 2K model, including all output gates.
    from nanochat.engine import KVCache
    from nanochat import flash_attention
    flash_attention.USE_FA3=False;torch.backends.cuda.enable_math_sdp(True)
    m.eval()
    with torch.no_grad():
        x,_=data.batch('dev',0,1);changed=x.clone();changed[:,1024:]=(changed[:,1024:]+7)%32768
        full=m(x);altered=m(changed)
        torch.testing.assert_close(full[:,:1024],altered[:,:1024],rtol=0,atol=0)
        del altered
        cache=KVCache(1,4,2048,128,8,'cuda',torch.bfloat16)
        parts=[];start=0
        for size in (511,1,513,1023):
            parts.append(m(x[:,start:start+size],kv_cache=cache));start+=size
        got=torch.cat(parts,1)
        rel=(got-full).norm()/full.norm()
        peak=(got-full).abs().max()/full.abs().max()
        assert rel<.008 and peak<.032,(rel.item(),peak.item())
        assert cache.get_pos()==2048
        result.update(causal_cache_passed=True,trained_cache_relative_error=rel.item(),trained_cache_peak_error=peak.item())
    json_write(a.output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
