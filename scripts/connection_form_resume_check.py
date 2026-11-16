"""Audit actual CLI exit85/restore across a Form, including fixed-gradient update."""
import argparse
import json
from pathlib import Path
import sys

import torch
from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
from nanochat.form import FormController
from nanochat.packed_training import PackedData, json_write, restore_rng, snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.directory)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
    from test_connection_training_resume import equal_tree
    configure_attention('fa2');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(root/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    assert state['step']==11 and state['form']['completed']==1 and state['form']['window_steps']==1
    rows=[json.loads(s) for s in (root/'steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,12))
    assert state['run_id']=='form-local-debug' and state['protocol']['form']['schedule'][0]==10
    assert state['form']['history'][0]['step']==10 and state['form']['history'][0]['next_train_ce']==rows[-1]['loss']
    pairs=[]
    for _ in range(2):
        c=ConnectionConfig.from_checkpoint(state['config'])
        model=build_connection_model(c).prepare(compile_blocks=True);opt,wd=optimizer_for(model)
        form=FormController(model,state['protocol']['steps'],state['protocol']['seed'])
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);form.load_state_dict(state['form'],11)
        equal_tree(model.state_dict(),state['model']);equal_tree(opt.state_dict(),state['optimizer'])
        equal_tree(form.state_dict(),state['form']);pairs.append((model,opt,wd,form))
    restore_rng(state['rng'])
    transport=root/'transport.pt'
    torch.save(snapshot(pairs[0][0],pairs[0][1],11,form=pairs[0][3].state_dict(),next_data_row=176),transport)
    loaded=torch.load(transport,map_location='cpu',weights_only=False)
    for key in ('model','optimizer','rng','form'):equal_tree(loaded[key],state[key])
    assert loaded['next_data_row']==176
    for model,opt,wd,form in pairs:
        for i,p in enumerate(model.parameters()):p.grad=torch.full_like(p,(i+1)*1e-6)
        schedule(opt,11,305176,wd);opt.step()
    equal_tree(pairs[0][0].state_dict(),pairs[1][0].state_dict())
    equal_tree(pairs[0][1].state_dict(),pairs[1][1].state_dict())
    # Best checkpoint may have an earlier, different shape; it must build too.
    best=torch.load(root/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    older=build_connection_model(ConnectionConfig.from_checkpoint(best['config'])).prepare()
    older.load_state_dict(best['model'])
    assert sum(older.config.ffn_widths)==sum(state['config']['ffn_widths'])==16384
    result=dict(complete=True,formal_training=False,step=11,exit85=True,next_data_row=176,
        widths=state['config']['ffn_widths'],form_completed=1,window_steps=1,
        checks=['CLI steps 1..11','same run/source/protocol','Form boundary not repeated',
            'dynamic model and optimizer shapes','exact EMA and RNG transport','isolated Form RNG',
            'bitwise fixed-gradient next update','load earlier dev-best architecture'])
    json_write(a.output,result);transport.unlink();print('FORM_RESUME_PASSED',flush=True)


if __name__=='__main__':main()
