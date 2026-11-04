"""GPU formula gate, same-device full-step profiles, and real process restart."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import torch
from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.packed_training import PackedData, seed_all, train_step, json_write


def profile(mode, out, kernel="original"):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    configure_attention('sdpa_cudnn');seed_all(42)
    model=ConnectionModel(ConnectionConfig(attention_statistic=mode, attention_kernel=kernel)).prepare(compile_blocks=True)
    opt,wd=optimizer_for(model);data=PackedData('../data')
    times=[];losses=[]
    for step in range(50):
        schedule(opt,step,305176,wd)
        if step==20:
            gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize();tick=time.perf_counter()
        loss=train_step(model,opt,data,step,microbatch=16)
        torch.cuda.synchronize();duration=time.perf_counter()-tick
        if step>=20:times.append(duration)
        losses.append(loss)
        if step in (0,19,49):print(mode,step+1,loss,duration,flush=True)
    result=dict(mode=mode,attention_kernel=kernel,mean_seconds=statistics.mean(times),samples_seconds=times,
        tokens_per_second=32768/statistics.mean(times),microbatch=16,grad_accum=1,
        warmup_steps=20,timed_steps=30,schedule_steps=305176,losses=losses,
        parameters=sum(p.numel() for p in model.parameters()),gpu=torch.cuda.get_device_name(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if mode=='variance_third':
        # Scale probe is outside all timing/peak-memory samples. Observe actual
        # 50-update activations and last-step parameter gradients, without norm.
        rows=[];handles=[];model.compiled_steps=None
        for layer,b in enumerate(model.blocks):
            for name in ('c_proj','c_stat','c_third'):
                module=getattr(b.attn,name)
                def capture(m,args,output,layer=layer,name=name):
                    x=args[0].detach().float();y=output.detach().float()
                    rows.append(dict(layer=layer,branch=name,input_rms=float(x.square().mean().sqrt()),
                        input_max=float(x.abs().max()),first_token_max=float(x[:,0].abs().max()),
                        output_rms=float(y.square().mean().sqrt()),
                        weight_gradient_norm=float(m.weight.grad.float().norm())))
                handles.append(module.register_forward_hook(capture))
        with torch.no_grad():model(data.batch('train',0,1)[0])
        for h in handles:h.remove()
        result['branch_scales_after_50_updates']=rows
    if mode=='variance_support':
        # Observe support/gate values after 50 real updates, outside timing and
        # peak-memory measurements. Hooks run only in this eager diagnostic.
        rows=[];handles=[];model.compiled_steps=None
        for layer,b in enumerate(model.blocks):
            def capture(m,args,output,layer=layer):
                logn=args[1].detach().float();g=m.coefficients(logn).detach()
                visible=torch.arange(1,logn.shape[1]+1,device=logn.device)[None,:,None]
                rows.append(dict(layer=layer,log_support_min=float(logn.min()),log_support_max=float(logn.max()),
                    first_token_log_support_max=float(logn[:,0].abs().max()),
                    support_visible_excess=float((logn.exp()-visible).max()),
                    gate_min=float(g.min()),gate_max=float(g.max()),gate_mean=float(g.mean()),
                    a=m.a.detach().tolist(),b=m.b.detach().tolist(),
                    a_gradient_norm=float(m.a.grad.norm()),b_gradient_norm=float(m.b.grad.norm())))
            handles.append(b.attn.support_gate.register_forward_hook(capture))
        with torch.no_grad():model(data.batch('train',0,1)[0])
        for h in handles:h.remove()
        result['support_after_50_updates']=rows
    json_write(out,result)
    assert result['peak_reserved_bytes'] < .9*22000*1024**2, result


def probe():
    """Record real backend support and FP32 cancellation against FP64 moments."""
    from nanochat.moment_attention import value_moments
    from torch.nn import functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.backends.cuda.matmul.allow_tf32=False;seed_all(42)
    xs=[torch.randn(2,256,4,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    q,k,v=xs;report={}
    for dtype in (torch.bfloat16,torch.float32):
        try:
            qs,ks,vs=[x.transpose(1,2).to(dtype).detach().requires_grad_() for x in xs]
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                y=F.scaled_dot_product_attention(qs,ks,torch.cat([vs,vs.square()],-1),is_causal=True)
                y.float().square().mean().backward()
            report[str(dtype)]=dict(supported_forward_backward=True,output_dtype=str(y.dtype))
        except (RuntimeError,NotImplementedError) as exc:
            report[str(dtype)]=dict(supported_forward_backward=False,error=str(exc))
    qf,kf,vf=[x.transpose(1,2).float() for x in xs]
    mask=torch.ones(256,256,device='cuda',dtype=torch.bool).tril()
    a=((qf@kf.transpose(-1,-2))/128**.5).masked_fill(~mask,-torch.inf).softmax(-1)
    raw=a@vf.square()-(a@vf).square()
    actual=value_moments(q,k,v,centered=True,recompute=False)
    reference=value_moments(q.double(),k.double(),v.double(),centered=True,recompute=False)
    error=(actual.double()-reference).abs()
    report['precision']=dict(raw_min=float(raw.min()),negative_fraction=float((raw<0).float().mean()),
        fp32_max_absolute_error=float(error.max()),fp32_relative_l2=float(error.norm()/reference.norm()),
        first_token_max=float(actual[:,0].abs().max()))
    assert error.max()<1e-5 and error.norm()/reference.norm()<3e-6,report
    report['selected']='compiled FP32 recomputed probabilities and moments; original cuDNN mean retained'
    return report


def execute(command, name, expected=0):
    log=Path('results')/(name+'.log')
    with log.open('w') as handle:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            handle.write(line);handle.flush();print(line,end='',flush=True)
        code=process.wait()
    assert code==expected,(name,code,expected)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile');parser.add_argument('--output');parser.add_argument('--kernel',default='original')
    args=parser.parse_args()
    if args.profile:return profile(args.profile,Path(args.output),args.kernel)
    Path('results').mkdir(exist_ok=True)
    assert not os.environ.get('WANDB_API_KEY')
    started=time.perf_counter()
    json_write('results/backend-probe.json',probe())
    execute([sys.executable,'-m','pytest','tests/test_moment_attention.py','tests/test_connection_training_resume.py',
        '-v','--junitxml=results/moment-tests.xml'],'tests')
    suites=ET.parse('results/moment-tests.xml').getroot().findall('testsuite')
    checks={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    assert checks==dict(tests=17,failures=0,errors=0,skipped=0),checks
    profiles=[]
    for repeat,order in enumerate([('none','second_moment','variance'),('variance','second_moment','none')]):
        for mode in order:
            name=f'profile-{repeat}-{mode}'
            execute([sys.executable,'-m','scripts.connection_moment_gate','--profile',mode,'--output','results/'+name+'.json'],name)
            profiles.append(json.loads(Path('results',name+'.json').read_text()))
    resumes={}
    for mode in ('second_moment','variance'):
        out=Path('results')/('resume-'+mode);ckpt=Path('../gate-checkpoints')/mode
        cmd=[sys.executable,'-m','scripts.connection_train','--variant','baseline',
            '--attention-statistic',mode,'--steps','50','--schedule-steps','305176',
            '--microbatch','16','--attention-backend','sdpa_cudnn','--matched-backbone-init',
            '--data-dir','../data','--output-dir',str(out),'--checkpoint-dir',str(ckpt),
            '--run-id','gate-'+mode,'--resume','auto','--segment-seconds','1800']
        execute(cmd+['--segment-steps','25'],'resume1-'+mode,85)
        assert json.loads((out/'progress.json').read_text())['step']==25
        execute(cmd,'resume2-'+mode)
        result=json.loads((out/'summary.json').read_text())
        assert result['complete'] and result['steps']==50 and result['training_tokens']==50*32768
        steps=[json.loads(line)['step'] for line in (out/'steps.jsonl').read_text().splitlines()]
        assert steps==list(range(1,51))
        config=json.loads((out/'config.json').read_text())
        assert config['model']['attention_statistic']==mode and config['logging_backend']=='local'
        assert config['protocol']['schedule_steps']==305176 and config['protocol']['grad_accum']==1
        resumes[mode]=dict(summary=result,config=config,exact_steps_1_to_50=True)
    assert not list(Path('results').rglob('*.wandb'))
    json_write('results/gate.json',dict(complete=True,source=json.loads(Path('source.json').read_text()),
        checks=checks,profiles=profiles,resumes=resumes,seconds=time.perf_counter()-started,
        formal_training=False,weights_returned=False))


if __name__=='__main__':main()
