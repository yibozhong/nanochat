"""GPU gate for post-step attention W_o NS, measured against the same baseline."""
import argparse,gc,json,os,statistics,sys,time
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for,schedule
from nanochat.attention_output_ns import attention_output_diagnostics
from nanochat.packed_training import PackedData,seed_all,train_step,evaluate,json_write
from scripts.connection_moment_gate import execute


def profile(mode,out):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;seed_all(42)
    model=ConnectionModel(ConnectionConfig(attention_kernel='fla_triton',attention_output_ns=mode=='output_ns')).prepare(compile_blocks=True)
    opt,wd=optimizer_for(model);data=PackedData('../data');times=[];losses=[]
    for step in range(50):
        schedule(opt,step,305176,wd)
        if step==20:gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize();tick=time.perf_counter()
        loss=train_step(model,opt,data,step)
        torch.cuda.synchronize();duration=time.perf_counter()-tick
        if step>=20:times.append(duration)
        losses.append(loss)
        if step in (0,19,49):print(mode,step+1,loss,duration,flush=True)
    allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    dev,tokens=evaluate(model,data);diagnostics=attention_output_diagnostics(model,data)
    result=dict(mode=mode,attention_kernel='fla_triton',mean_seconds=statistics.mean(times),samples_seconds=times,
        tokens_per_second=32768/statistics.mean(times),microbatch=16,grad_accum=1,
        warmup_steps=20,timed_steps=30,schedule_steps=305176,losses=losses,dev_ce=dev,dev_tokens=tokens,
        parameters=sum(p.numel() for p in model.parameters()),gpu=torch.cuda.get_device_name(),
        peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,diagnostics=diagnostics)
    json_write(out,result)
    assert reserved<.9*22000*1024**2,result
    if mode=='output_ns':
        assert all(row['weight']['scaled_orthogonality_error']<.2 for row in diagnostics['layers'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile');parser.add_argument('--output')
    args=parser.parse_args()
    if args.profile:return profile(args.profile,Path(args.output))
    Path('results').mkdir(exist_ok=True);assert not os.environ.get('WANDB_API_KEY')
    started=time.perf_counter()
    execute([sys.executable,'-m','pytest','tests/test_attention_output_ns.py','tests/test_triton_resume.py',
        '-v','--junitxml=results/output-ns-tests.xml'],'tests')
    suites=ET.parse('results/output-ns-tests.xml').getroot().findall('testsuite')
    checks={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    assert checks==dict(tests=17,failures=0,errors=0,skipped=0),checks
    profiles=[]
    for repeat,order in enumerate([('baseline','output_ns'),('output_ns','baseline')]):
        for mode in order:
            name=f'profile-{repeat}-{mode}'
            execute([sys.executable,'-m','scripts.connection_output_ns_gate','--profile',mode,'--output','results/'+name+'.json'],name)
            profiles.append(json.loads(Path('results',name+'.json').read_text()))
    out=Path('results/resume');ckpt=Path('../gate-checkpoints/output-ns')
    cmd=[sys.executable,'-m','scripts.connection_train','--variant','baseline','--attention-output-ns',
        '--attention-kernel','fla_triton','--steps','50','--schedule-steps','305176','--microbatch','16',
        '--attention-backend','sdpa_cudnn','--matched-backbone-init','--data-dir','../data',
        '--output-dir',str(out),'--checkpoint-dir',str(ckpt),'--run-id','gate-output-ns',
        '--resume','auto','--segment-seconds','1800']
    execute(cmd+['--segment-steps','25'],'resume1',85)
    assert json.loads((out/'progress.json').read_text())['step']==25
    execute(cmd,'resume2')
    result=json.loads((out/'summary.json').read_text());config=json.loads((out/'config.json').read_text())
    assert result['complete'] and result['steps']==50 and result['training_tokens']==50*32768
    rows=[json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,51))
    reports=[json.loads(line) for line in (out/'attention-output-spectra.jsonl').read_text().splitlines()]
    assert [r['step'] for r in reports]==[0,25,50]
    for report in reports[1:]:
        assert len(report['layers'])==8 and all(row['weight']['scaled_orthogonality_error']<.2 for row in report['layers'])
    assert config['model']['attention_output_ns'] and config['logging_backend']=='local'
    assert config['protocol']['schedule_steps']==305176 and config['protocol']['grad_accum']==1
    assert not list(Path('results').rglob('*.wandb')) and not list(Path('results').rglob('*.pt'))
    json_write('results/gate.json',dict(complete=True,source=json.loads(Path('source.json').read_text()),
        checks=checks,profiles=profiles,resume=dict(summary=result,config=config,exact_steps_1_to_50=True,
        projection_after_resume_verified=True,spectra=reports),seconds=time.perf_counter()-started,
        formal_training=False,weights_returned=False))


if __name__=='__main__':main()
