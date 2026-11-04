"""FA3 alternating attention-to-MLP replacement checks/profile/restart, without formal training."""
import argparse,gc,json,os,statistics,sys,time
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
from nanochat import flash_attention
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for,schedule
from nanochat.connection_experiment import configure_attention
from nanochat.connection_cost import model_cost
from nanochat.packed_training import PackedData,seed_all,train_step,evaluate,json_write
from scripts.connection_moment_gate import execute


def profile(mode,out):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;seed_all(42);configure_attention('fa3')
    model=ConnectionModel(ConnectionConfig(mlp_heavy=mode=='mlp_heavy')).prepare(compile_blocks=True)
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
    dev,tokens=evaluate(model,data)
    result=dict(mode=mode,attention_kernel='original',attention_backend='fa3',mean_seconds=statistics.mean(times),samples_seconds=times,
        tokens_per_second=32768/statistics.mean(times),microbatch=16,grad_accum=1,
        warmup_steps=20,timed_steps=30,schedule_steps=305176,losses=losses,dev_ce=dev,dev_tokens=tokens,
        parameters=sum(p.numel() for p in model.parameters()),gpu=torch.cuda.get_device_name(),
        peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,analytic_cost=model_cost(model.config,batch=16))
    json_write(out,result)
    assert reserved<.9*22000*1024**2,result


def backend_probe():
    configure_attention('fa3');assert flash_attention.USE_FA3
    xs=[torch.randn(1,2048,4,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    for window in ((-1,0),):
        flash_attention.flash_attn_func(*xs,causal=True,window_size=window).float().sum().backward()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        for window in ((-1,0),):
            flash_attention.flash_attn_func(*xs,causal=True,window_size=window).float().sum().backward()
    names=sorted(set(event.name for event in prof.events() if event.device_type==torch.autograd.DeviceType.CUDA))
    assert any('flash' in name.lower() for name in names),names
    json_write('results/backend-probe.json',dict(backend='fa3',module=flash_attention._fa3.__file__,
        revision=os.environ['NANOCHAT_FA3_REVISION'],windows=[[-1,0]],cuda_kernels=names))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile');parser.add_argument('--output')
    args=parser.parse_args()
    if args.profile:return profile(args.profile,Path(args.output))
    Path('results').mkdir(exist_ok=True);assert not os.environ.get('WANDB_API_KEY')
    started=time.perf_counter();backend_probe()
    execute([sys.executable,'-m','pytest','tests/test_mlp_heavy.py','tests/test_triton_moments.py','tests/test_triton_resume.py','tests/test_connection_training_resume.py',
        '-v','--junitxml=results/mlp-heavy-tests.xml'],'tests')
    suites=ET.parse('results/mlp-heavy-tests.xml').getroot().findall('testsuite')
    checks={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    assert checks==dict(tests=37,failures=0,errors=0,skipped=0),checks
    profiles=[]
    for repeat,order in enumerate([('baseline','mlp_heavy'),('mlp_heavy','baseline')]):
        for mode in order:
            name=f'profile-{repeat}-{mode}'
            execute([sys.executable,'-m','scripts.connection_mlp_heavy_gate','--profile',mode,'--output','results/'+name+'.json'],name)
            profiles.append(json.loads(Path('results',name+'.json').read_text()))
    out=Path('results/resume');ckpt=Path('../gate-checkpoints/mlp-heavy')
    cmd=[sys.executable,'-m','scripts.connection_train','--variant','baseline','--mlp-heavy',
        '--attention-kernel','original',
        '--steps','50','--schedule-steps','305176','--microbatch','16',
        '--attention-backend','fa3','--matched-backbone-init','--data-dir','../data',
        '--output-dir',str(out),'--checkpoint-dir',str(ckpt),'--run-id','gate-mlp-heavy',
        '--resume','auto','--segment-seconds','1800']
    execute(cmd+['--segment-steps','25'],'resume1',85)
    assert json.loads((out/'progress.json').read_text())['step']==25
    execute(cmd,'resume2')
    result=json.loads((out/'summary.json').read_text());config=json.loads((out/'config.json').read_text())
    assert result['complete'] and result['steps']==50 and result['training_tokens']==50*32768
    rows=[json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,51))
    assert config['model']['mlp_heavy'] and config['logging_backend']=='local'
    assert config['parameters']==62914560
    assert config['protocol']['schedule_steps']==305176 and config['protocol']['grad_accum']==1
    assert not list(Path('results').rglob('*.wandb')) and not list(Path('results').rglob('*.pt'))
    json_write('results/gate.json',dict(complete=True,source=json.loads(Path('source.json').read_text()),
        checks=checks,profiles=profiles,resume=dict(summary=result,config=config,exact_steps_1_to_50=True),
        seconds=time.perf_counter()-started,formal_training=False,weights_returned=False))


if __name__=='__main__':main()
