"""Bounded same-GPU vanilla/Form comparison on the actual packed recipe."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import torch
from nanochat.connection_cost import model_cost
from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
from nanochat.form import FormController
from nanochat.packed_training import PackedData, evaluate, json_write, seed_all, train_step


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    configure_attention('fa2');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    data=PackedData(a.data_dir);models={};timings={};losses={};events=[]
    for label,mode in [('baseline','none'),('form','taylor')]:
        seed_all(42)
        model=build_connection_model(ConnectionConfig(streams=1,form_mode=mode)).prepare(compile_blocks=True)
        opt,wd=optimizer_for(model)
        controller=FormController(model,1056) if mode!='none' else None
        models[label]=[model,opt,wd,controller];timings[label]=[];losses[label]=[]
    left,right=models['baseline'][0],models['form'][0]
    assert all(torch.equal(p,dict(right.named_parameters())[n]) for n,p in left.named_parameters())
    assert model_cost(left.config)==model_cost(right.config)
    torch.cuda.reset_peak_memory_stats();last_form=0
    with (out/'steps.jsonl').open('w',buffering=1) as log:
        for step in range(256):
            for label in (('baseline','form') if step%2==0 else ('form','baseline')):
                m,opt,wd,form=models[label];schedule(opt,step,305176,wd)
                torch.cuda.synchronize();start=time.perf_counter()
                loss=train_step(m,opt,data,step,microbatch=16,seed=42,form=form)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                row=dict(model=label,step=step+1,loss=loss,seconds=elapsed)
                log.write(json.dumps(row)+'\n');losses[label].append(loss)
                # Symmetric warmup exclusion: both models recompile after reset.
                if step>=16 and (step<96 or 112<=step<192 or step>=208):
                    timings[label].append(elapsed)
                if form is not None and form.due(step+1):
                    torch.cuda.synchronize();begin=time.perf_counter()
                    opt,report=form.apply(opt,step+1,loss);models[label][1]=opt
                    torch.cuda.synchronize();seconds=time.perf_counter()-begin
                    events.append(dict(**report['event'],structural_update_seconds=seconds))
                    json_write(out/f'form-{form.completed}.json',report);last_form=step+1
                if step<3 or (step+1)%32==0:print(json.dumps(row),flush=True)
    result=dict(complete=True,formal_training=False,gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,attention_backend='fa2',
        steps_per_model=256,tokens_per_model=256*32768,microbatch=16,seed=42,
        models={},events=events,combined_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        data_manifest=data.manifest,extra_scoring_forward_backward_passes=0)
    for label,(m,opt,wd,form) in models.items():
        assert losses[label][-1]<losses[label][0]
        result['models'][label]=dict(config=asdict(m.config),parameters=sum(p.numel() for p in m.parameters()),
            first_loss=losses[label][0],last_loss=losses[label][-1],dev_ce=evaluate(m,data)[0],
            mean_step_seconds=statistics.mean(timings[label]),median_step_seconds=statistics.median(timings[label]),
            measured_steps=len(timings[label]),cost=model_cost(m.config,batch=16))
    assert models['form'][3].completed==2
    result['step_time_ratio']=result['models']['form']['mean_step_seconds']/result['models']['baseline']['mean_step_seconds']
    json_write(out/'summary.json',result)
    assert result['step_time_ratio']<2., 'Inspect unexpected >2x Form overhead before formal training'
    print('FORM_PROFILE_PASSED '+json.dumps(dict(ratio=result['step_time_ratio'],events=len(events))),flush=True)


if __name__=='__main__':main()
