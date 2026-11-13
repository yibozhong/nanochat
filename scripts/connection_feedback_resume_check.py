"""Exact checkpoint/next-update checks, independent of nondeterministic backward.

Production FA2/CCE backward can differ on repeated identical inputs. Compare
transported state and the next fixed-gradient optimizer update bitwise; report
independent trajectory differences separately instead of loosening tolerance.
"""
import argparse
from pathlib import Path
import sys

import torch

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
from nanochat.packed_training import PackedData, json_write, restore_rng, snapshot


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--resumed',required=True)
    p.add_argument('--continuous',required=True)
    p.add_argument('--data-dir',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
    from test_connection_training_resume import equal_tree
    configure_attention('fa2');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    left=torch.load(Path(a.resumed)/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    right=torch.load(Path(a.continuous)/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    assert left['step']==right['step']==3
    for key in ('rng','config','protocol','source','run_id','data_manifest'):
        equal_tree(left[key],right[key])
    import json
    rows=[json.loads(line) for line in (Path(a.resumed)/'steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==[1,2,3]
    c=ConnectionConfig.from_checkpoint(left['config'])
    model=build_connection_model(c).prepare(compile_blocks=True)
    opt,wd=optimizer_for(model)
    model.load_state_dict(left['model']);opt.load_state_dict(left['optimizer'])
    equal_tree(model.state_dict(),left['model'])
    equal_tree(opt.state_dict(),left['optimizer'])
    restore_rng(left['rng'])
    transport=Path(a.resumed)/'transport-test.pt'
    torch.save(snapshot(model,opt,left['step'],next_data_row=left['step']*16),transport)
    loaded=torch.load(transport,map_location='cpu',weights_only=False)
    restored=build_connection_model(c).prepare(compile_blocks=True)
    other,other_wd=optimizer_for(restored)
    restored.load_state_dict(loaded['model']);other.load_state_dict(loaded['optimizer'])
    equal_tree(model.state_dict(),restored.state_dict())
    equal_tree(opt.state_dict(),other.state_dict())
    assert loaded['next_data_row']==48 and wd==other_wd
    equal_tree(loaded['rng'],left['rng'])
    data=PackedData(a.data_dir)
    equal_tree(data.manifest,left['data_manifest'])
    x,y=data.batch('train',loaded['next_data_row'],16)
    passes=model.training_passes(left['protocol']['seed'],left['step'])
    restore_rng(loaded['rng']);model.zero_grad(set_to_none=True)
    loss=model(x,y,passes=passes);loss.backward()
    gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradients.values())
    for target,optimizer in ((model,opt),(restored,other)):
        optimizer.zero_grad(set_to_none=True)
        for n,p in target.named_parameters():
            p.grad=gradients[n].clone() if n in gradients else None
        schedule(optimizer,left['step'],left['protocol']['schedule_steps'],wd)
        optimizer.step()
    equal_tree(model.state_dict(),restored.state_dict())
    equal_tree(opt.state_dict(),other.state_dict())
    differences=[]
    for name,value in left['model'].items():
        diff=(value.float()-right['model'][name].float()).abs()
        if diff.max():
            differences.append(dict(name=name,max_abs=diff.max().item(),
                relative_l2=(diff.norm()/right['model'][name].float().norm().clamp_min(1e-20)).item()))
    result=dict(complete=True,formal_training=False,step=3,exit85=True,next_data_row=48,
        checks=['CLI step sequence','run ID','source','config','protocol','data manifest','RNG',
                'exact state load','exact state roundtrip','bitwise next fixed-gradient update'],
        next_passes=passes,next_loss=loss.item(),
        independent_trajectory_bitwise_equal=not differences,
        independent_trajectory_differences=differences,
        note='Independent production backward is nondeterministic. No tolerance is used for checkpoint state or fixed-gradient update checks.')
    json_write(a.output,result);transport.unlink()
    print('EXACT_RESUME_CHECK_PASSED',flush=True)


if __name__=='__main__':main()
