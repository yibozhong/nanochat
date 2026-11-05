"""Frozen-weight speed measurements; no optimizer, updates, W&B or checkpoints."""
import argparse,gc,hashlib,json,statistics,time
from pathlib import Path
import torch
from scripts.stream_scaling_model import ScalingModel,GeneralConnection,Projection
from nanochat.nvidia_connections import NvidiaHyperConnection
from nanochat.stream_mlp_connections import StreamMLPHyperConnection
from nanochat.connection_experiment import configure_attention
from nanochat.packed_training import PackedData


def measure(fn):
    for _ in range(8):fn()
    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    samples=[]
    for _ in range(20):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();value=fn();b.record();b.synchronize();samples.append(a.elapsed_time(b));del value
    return dict(mean_ms=statistics.mean(samples),median_ms=statistics.median(samples),
        samples_ms=samples,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved())


def flops(n,kind,b,t=2048,d=512,depth=8,v=32768):
    routing=2*(n*d)*(2*n+(n*n if kind=='mhc' else 0))
    mixing=2*d*n*n if kind=='mhc' else 4*d*n*(4*n)
    connectors=2*depth*(routing+mixing+4*n*d)
    head=2*n*n*d+2*n*d
    backbone=24*depth*d*d+2*d*v+2*d*depth*(t+1)
    return dict(connector_per_token_forward=routing+mixing+4*n*d,
        routing_per_token_forward=routing,mixing_per_token_forward=mixing,
        model_forward_matrix_flops=b*t*(backbone+connectors+head),
        note='FMA2; causal useful attention pairs; excludes norms/activations/Sinkhorn/precision padding/recomputation. No measured hardware instruction count.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=('mhc','mlp'),required=True)
    p.add_argument('--streams',type=int,choices=(4,8,16,32),required=True)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--scope',choices=('micro','model'),required=True)
    p.add_argument('--anchor',action='store_true');p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(42)
    torch._dynamo.config.recompile_limit=64
    backend=configure_attention('hardware_auto');torch.backends.cuda.matmul.allow_tf32=True
    props=torch.cuda.get_device_properties(0)
    report=dict(complete=False,status='started',speed_only=True,parameter_updates=0,training_tokens=0,
        kind=a.kind,streams=a.streams,mlp_inner=4*a.streams if a.kind=='mlp' else None,
        batch=a.batch,sequence_length=2048,width_per_stream=512,residual_width=a.streams*512,
        scope=a.scope,anchor=a.anchor,gpu=props.name,gpu_uuid=str(props.uuid),device_bytes=props.total_memory,
        attention_backend=backend,source=json.loads(Path('source.json').read_text()),
        implementation='production4_anchor' if a.anchor else 'generalized_matched_backend',
        flops=flops(a.streams,a.kind,a.batch))
    try:
        if a.scope=='model':
            model=ScalingModel(a.kind,a.streams,anchor=a.anchor)
            data=PackedData('../data');ids,targets=data.batch('train',0,a.batch)
            report['input_sha256']=hashlib.sha256(ids.cpu().numpy().tobytes()+targets.cpu().numpy().tobytes()).hexdigest()
            params=tuple(model.parameters());report['parameters']=sum(x.numel() for x in params)
            versions=[x._version for x in params]
            def forward():
                with torch.no_grad():return model(ids,targets)
            def backward():
                loss=model(ids,targets)
                return loss,torch.autograd.grad(loss,params)
            loss,grads=backward()
            assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads)
            report['initial_loss_sanity_only']=loss.item();del loss,grads
        else:
            n=a.streams;d=512
            if a.anchor:
                model=(NvidiaHyperConnection(d,n,0) if a.kind=='mhc' else
                       StreamMLPHyperConnection(d,n,0,compact_routing=True,stream_mlp_kernel='rowwise')).cuda()
                model.reset_parameters()
            else:model=GeneralConnection(d,n,a.kind).cuda()
            x=torch.randn(a.batch,2048,d,n,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            y=torch.randn(a.batch,2048,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            gb=torch.randn_like(y);go=torch.randn_like(x);params=tuple(model.parameters());versions=[x._version for x in params]
            def call():
                b,s,m=model.read(x);return b,model.write(y,s,m)
            def forward():
                with torch.no_grad():return call()
            def backward():
                values=call();return values,torch.autograd.grad(values,(x,y,*params),(gb,go))
            if not a.anchor:
                flat=x.reshape(-1,d*n);gh=torch.randn(flat.shape[0],model.phi.shape[0],device='cuda');gm=torch.randn(flat.shape[0],device='cuda')*.001
                def projection():
                    raw,ms=Projection.apply(flat,model.phi)
                    return (raw,ms),torch.autograd.grad((raw,ms),(x,model.phi),(gh,gm))
                report['projection_forward_backward']=measure(projection)
                del gh,gm
        report['forward']=measure(forward)
        report['forward_backward']=measure(backward)
        report['repeat_forward_backward']=measure(backward)
        assert versions==[x._version for x in params]
        report['tokens_per_second_forward_backward']=a.batch*2048/(report['forward_backward']['mean_ms']/1000)
        report.update(complete=True,status='measured',weights_unchanged=True)
    except torch.OutOfMemoryError as exc:
        report.update(complete=True,status='oom',error=str(exc)[:1500],peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report.get(k) for k in ('kind','streams','scope','anchor','batch','status','tokens_per_second_forward_backward')}),flush=True)


if __name__=='__main__':main()
