"""Innovation residual: GPU correctness, matched full-step cost and real restart."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
import xml.etree.ElementTree as ET

import torch
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for,schedule
from nanochat.packed_training import PackedData,seed_all,train_step,evaluate,json_write
from scripts.connection_moment_gate import execute


def profile(mode,out):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;seed_all(42)
    model=ConnectionModel(ConnectionConfig(attention_kernel='fla_triton',innovation_residual=mode=='innovation')).prepare(compile_blocks=True)
    opt,wd=optimizer_for(model);data=PackedData('../data');times=[];losses=[];metrics={}
    for step in range(50):
        schedule(opt,step,305176,wd)
        if step==20:gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize();tick=time.perf_counter()
        loss=train_step(model,opt,data,step,metrics=metrics)
        torch.cuda.synchronize();duration=time.perf_counter()-tick
        if step>=20:times.append(duration)
        losses.append(loss)
        if step in (0,19,49):print(mode,step+1,loss,duration,metrics,flush=True)
    allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    report={} if mode=='innovation' else None
    dev,tokens=evaluate(model,data,innovation_report=report)
    result=dict(mode=mode,attention_kernel='fla_triton',mean_seconds=statistics.mean(times),samples_seconds=times,
        tokens_per_second=32768/statistics.mean(times),microbatch=16,grad_accum=1,
        warmup_steps=20,timed_steps=30,schedule_steps=305176,losses=losses,dev_ce=dev,dev_tokens=tokens,
        parameters=sum(p.numel() for p in model.parameters()),gpu=torch.cuda.get_device_name(),
        peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,innovation=report)
    json_write(out,result)
    assert reserved < .9*22000*1024**2,result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile');parser.add_argument('--output')
    args=parser.parse_args()
    if args.profile:return profile(args.profile,Path(args.output))
    Path('results').mkdir(exist_ok=True);assert not os.environ.get('WANDB_API_KEY')
    started=time.perf_counter()
    execute([sys.executable,'-m','pytest','tests/test_innovation_residual.py','tests/test_triton_resume.py',
             '-v','--junitxml=results/innovation-tests.xml'],'tests')
    suites=ET.parse('results/innovation-tests.xml').getroot().findall('testsuite')
    checks={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    assert checks==dict(tests=19,failures=0,errors=0,skipped=0),checks
    profiles=[]
    for repeat,order in enumerate([('baseline','innovation'),('innovation','baseline')]):
        for mode in order:
            name=f'profile-{repeat}-{mode}'
            execute([sys.executable,'-m','scripts.connection_innovation_gate','--profile',mode,'--output','results/'+name+'.json'],name)
            profiles.append(json.loads(Path('results',name+'.json').read_text()))
    out=Path('results/resume');ckpt=Path('../gate-checkpoints/innovation')
    cmd=[sys.executable,'-m','scripts.connection_train','--variant','baseline','--innovation-residual',
        '--attention-kernel','fla_triton','--steps','50','--schedule-steps','305176','--microbatch','16',
        '--attention-backend','sdpa_cudnn','--matched-backbone-init','--data-dir','../data',
        '--output-dir',str(out),'--checkpoint-dir',str(ckpt),'--run-id','gate-innovation',
        '--resume','auto','--segment-seconds','1800']
    execute(cmd+['--segment-steps','25'],'resume1',85)
    assert json.loads((out/'progress.json').read_text())['step']==25
    execute(cmd,'resume2')
    result=json.loads((out/'summary.json').read_text());config=json.loads((out/'config.json').read_text())
    assert result['complete'] and result['steps']==50 and result['training_tokens']==50*32768
    rows=[json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,51))
    for row in rows:
        assert abs(row['innovation/beta']-min((row['step']-1)/39,1.))<1e-7
    reports=[json.loads(line) for line in (out/'innovation.jsonl').read_text().splitlines()]
    assert [r['step'] for r in reports]==[0,25,50]
    assert all(len(r['branches'])==16 for r in reports)
    assert config['model']['innovation_residual'] and config['logging_backend']=='local'
    assert config['protocol']['schedule_steps']==305176 and config['protocol']['grad_accum']==1
    assert not list(Path('results').rglob('*.wandb')) and not list(Path('results').rglob('*.pt'))
    json_write('results/gate.json',dict(complete=True,source=json.loads(Path('source.json').read_text()),
        checks=checks,profiles=profiles,resume=dict(summary=result,config=config,exact_steps_1_to_50=True,
        beta_schedule_restored=True,innovation_reports=reports),seconds=time.perf_counter()-started,
        formal_training=False,weights_returned=False))


if __name__=='__main__':main()
