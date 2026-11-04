"""Read-only investigation of finite NS error and compiled BF16 rounding."""
import json
from pathlib import Path
import torch
from nanochat.attention_output_ns import scaled_ns_projection,spectrum_summary
from nanochat.optim import polar_express_coeffs
from nanochat.packed_training import json_write,seed_all
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for,schedule


def explicit_pointwise(weights):
    target=weights.float().norm(dim=(-2,-1),keepdim=True)
    x=weights.bfloat16()
    x=(x.float()/(x.float().norm(dim=(-2,-1),keepdim=True)*1.01+1e-6)).bfloat16()
    for a,b,c in polar_express_coeffs:
        gram=x@x.mT
        squared=gram@gram
        polynomial=(b*gram.float()+c*squared.float()).bfloat16()
        update=polynomial@x
        x=(a*x.float()+update.float()).bfloat16()
    result=x.float()
    return result*(target/result.norm(dim=(-2,-1),keepdim=True).clamp_min(1e-30))


def spectral_oracle(weights):
    """FP64 scalar polynomial applied to singular values, not matrix iteration."""
    u,s,vh=torch.linalg.svd(weights.double(),full_matrices=False)
    norm=s.norm(dim=-1,keepdim=True)
    x=s/(norm*1.01+1e-6)
    for a,b,c in polar_express_coeffs:x=a*x+b*x**3+c*x**5
    x=x*(norm/x.norm(dim=-1,keepdim=True))
    return (u*x.unsqueeze(-2))@vh


@torch.no_grad()
def inspect(name,weights,compiled_old,compiled_explicit):
    variants=dict(input=weights,eager=scaled_ns_projection(weights),compiled=compiled_old(weights),
        explicit_eager=explicit_pointwise(weights),explicit_compiled=compiled_explicit(weights),
        fp64_scalar=spectral_oracle(weights))
    rows=[]
    for index in range(len(weights)):
        ref=variants['fp64_scalar'][index];row=dict(name=name,index=index,variants={})
        for mode,matrices in variants.items():
            w=matrices[index].double();s=torch.linalg.svdvals(w)
            row['variants'][mode]=dict(**spectrum_summary(s),relative_l2_to_fp64=float((w-ref).norm()/ref.norm()),
                relative_norm_change=float(w.norm()/weights[index].double().norm()-1))
        for a,b in (('eager','compiled'),('explicit_eager','explicit_compiled')):
            left=variants[a][index].double();right=variants[b][index].double()
            row[a+'_vs_'+b]=float((left-right).norm()/left.norm())
        rows.append(row)
    print(json.dumps(rows),flush=True)
    return rows


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;seed_all(42)
    compiled_old=torch.compile(scaled_ns_projection,fullgraph=True,dynamic=False)
    compiled_explicit=torch.compile(explicit_pointwise,fullgraph=True,dynamic=False)
    rows=[]
    for d in (128,512):
        torch.manual_seed(5)
        w=torch.randn(2,d,d,device='cuda');w[0]=torch.diag(torch.linspace(.1,2.,d,device='cuda'))
        rows.extend(inspect('synthetic-'+str(d),w,compiled_old,compiled_explicit))
    seed_all(42)
    model=ConnectionModel(ConnectionConfig(attention_kernel='fla_triton',depth=2,width=128,
        heads=1,vocab_size=256,sequence_length=128)).prepare()
    opt,wd=optimizer_for(model);x=torch.randint(256,(2,128),device='cuda');y=(x+1)%256
    model(x,y).backward();schedule(opt,0,305176,wd);opt.step()
    weights=torch.stack([b.attn.c_proj.weight.detach() for b in model.blocks])
    rows.extend(inspect('actual-first-update',weights,compiled_old,compiled_explicit))
    Path('results').mkdir(exist_ok=True)
    json_write('results/ns-numerics.json',dict(complete=True,source=json.loads(Path('source.json').read_text()),
        gpu=torch.cuda.get_device_name(),rows=rows,formal_training=False,weights_returned=False))


if __name__=='__main__':main()
