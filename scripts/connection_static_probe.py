"""One bounded forward/backward scale probe at initialization or saved step128."""
import argparse,json
from pathlib import Path
import torch
from nanochat.connection_model import ConnectionConfig, ConnectionModel, STATIC_VARIANTS
from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.packed_training import PackedData, seed_all


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=STATIC_VARIANTS,required=True)
    p.add_argument('--checkpoint');p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    configure_attention('fa3');seed_all(42)
    cfg=ConnectionConfig(variant=a.variant)
    m=ConnectionModel(cfg).prepare();match_vanilla_initialization(m,42)
    step=0
    if a.checkpoint:
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
        assert ConnectionConfig.from_checkpoint(state['config'])==cfg
        m.load_state_dict(state['model']);step=state['step'];del state
    captured={}
    for i,b in enumerate(m.blocks):
        for kind in ('attn_connection','mlp_connection'):
            conn=getattr(b,kind);original=conn.write;name=f'{i}/{kind}'
            def wrapped(y,x,post,original=original,name=name):
                out=original(y,x,post);out.retain_grad()
                captured[name]=(x,y,out);return out
            conn.write=wrapped
    x,y=PackedData('../data').batch('dev',0,1)
    loss=m(x,y);loss.backward()
    def rms(t):return t.detach().float().square().mean().sqrt().item()
    rows={k:dict(input_rms=rms(x),branch_rms=rms(y),output_rms=rms(out),output_gradient_rms=rms(out.grad)) for k,(x,y,out) in captured.items()}
    finite_params=all(torch.isfinite(p).all().item() for p in m.parameters())
    finite_grads=all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in m.parameters())
    assert torch.isfinite(loss) and finite_params and finite_grads
    result=dict(complete=True,variant=a.variant,step=step,probe_tokens=y.numel(),ce=loss.item(),layers=rows,
                parameters_finite=finite_params,gradients_finite=finite_grads,
                maximum_output_rms=max(v['output_rms'] for v in rows.values()),
                scope='One fixed2048-token dev sequence, no optimizer update; scale/gradient diagnostics, not a quality comparison')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
