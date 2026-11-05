"""Read-only 2K activation/gradient scale probe at init or a saved checkpoint."""
import argparse,json
from pathlib import Path
import torch
from nanochat.connection_model import ConnectionConfig,ConnectionModel
from nanochat.connection_experiment import configure_attention,match_vanilla_initialization
from nanochat.packed_training import PackedData,seed_all


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    configure_attention('fa3');seed_all(42)
    cfg=ConnectionConfig(variant='mhc_stream_mlp');m=ConnectionModel(cfg).prepare();match_vanilla_initialization(m,42)
    step=0
    if a.checkpoint:
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
        assert ConnectionConfig.from_checkpoint(saved['config'])==cfg
        m.load_state_dict(saved['model']);step=saved['step'];del saved
    captured={}
    for i,b in enumerate(m.blocks):
        for kind in ('attn_connection','mlp_connection'):
            conn=getattr(b,kind);original=conn.write;name=f'{i}/{kind}'
            def wrapped(y,x,post,original=original,name=name):
                out=original(y,x,post);out.retain_grad();captured[name]=(x,y,out);return out
            conn.write=wrapped
    x,labels=PackedData('../data').batch('dev',0,1)
    loss=m(x,labels);loss.backward()
    def rms(t):return t.detach().float().square().mean().sqrt().item()
    rows={k:dict(input_rms=rms(x),branch_rms=rms(y),output_rms=rms(o),output_gradient_rms=rms(o.grad)) for k,(x,y,o) in captured.items()}
    params_finite=all(torch.isfinite(p).all().item() for p in m.parameters())
    grads_finite=all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in m.parameters())
    assert torch.isfinite(loss) and params_finite and grads_finite
    result=dict(complete=True,variant=cfg.variant,step=step,probe_tokens=labels.numel(),ce=loss.item(),layers=rows,
        parameters_finite=params_finite,gradients_finite=grads_finite,maximum_output_rms=max(v['output_rms'] for v in rows.values()),
        scope='Fixed one2K dev sequence; no optimizer update; short diagnostics do not establish long-term stability')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
