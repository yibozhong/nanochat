"""Locate the first nonfinite optimizer operation in a bounded diagnostic replay."""
import gc
import json
from pathlib import Path

import torch
from nanochat import optim
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.connection_experiment import configure_attention
from nanochat.packed_training import PackedData, seed_all, restore_rng
from scripts.connection_tokenres_diagnostic import stats, write


@torch.no_grad()
def shadow(args, normalize_input=False):
    grads, params, momentum, second = (a.clone() for a in args[:4])
    mt, lr, wd, bt, steps, red_dim = args[4:]
    stages = {}
    def record(name, x):
        stages[name] = stats(x)
        return x
    momentum.lerp_(grads, 1-mt.to(grads.dtype))
    g = record('nesterov', grads.lerp_(momentum, mt.to(grads.dtype)))
    # A scale-invariance counterfactual inside the orthogonalizer ONLY.
    # Original optimizer call above remains unchanged.
    if normalize_input:
        g = g/g.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    x = record('cast_bf16', g.bfloat16())
    target = record('target_norm_fp32', x.float().norm(dim=(-2, -1), keepdim=True)/(x.size(-2)**.5))
    row = record('row_norm_fp32', x.float().norm(dim=-1, keepdim=True).clamp_min(1e-6))
    x = record('row_equilibrated', x*(target/row).to(x.dtype))
    denom = record('polar_denominator', x.norm(dim=(-2, -1), keepdim=True)*1.01+1e-6)
    x = record('polar_normalized', x/denom)
    for i, (a, b, c) in enumerate(optim.polar_express_coeffs[:steps]):
        if g.size(-2)>g.size(-1):
            A = x.mT@x; B = b*A+c*(A@A); x = a*x+x@B
        else:
            A = x@x.mT; B = b*A+c*(A@A); x = a*x+B@x
        record('polar_'+str(i), x)
    g = x.to(params.dtype)
    g = record('muon_plus', g*((min(g.size(-2), g.size(-1))**.5)/g.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)).to(g.dtype))
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True)*g.size(red_dim)).sqrt()
    second.lerp_(v_mean.to(second.dtype), 1-bt.to(g.dtype))
    step_size = second.clamp_min(1e-10).rsqrt()
    new_norm = ((v_mean*g.size(red_dim))*step_size.float().square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    g = record('variance_scaled', g*(step_size*(v_norm/new_norm.clamp_min(1e-10))).to(g.dtype))
    params.sub_(lr.to(g.dtype)*g+lr.to(g.dtype)*wd.to(g.dtype)*params*((g*params)>=0))
    record('parameters_after', params)
    return stages


def main():
    out = Path('results/optimizer-probe'); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    configure_attention('fa3'); seed_all(42)
    data = PackedData('../data')
    state = torch.load('../checkpoints/latest.pt', map_location='cpu', weights_only=False)
    model = ConnectionModel(ConnectionConfig.from_checkpoint(state['config'])).prepare(compile_blocks=True)
    model.load_state_dict(state['model']); opt, wd = optimizer_for(model)
    opt.load_state_dict(state['optimizer']); restore_rng(state['rng']); start = state['step']
    del state; gc.collect()
    original = optim.muon_step_fused
    events = []
    names = {id(p): n for n, p in model.named_parameters()}
    groups = {tuple(g['params'][0].shape): [names[id(p)] for p in g['params']]
              for g in opt.param_groups if g['kind']=='muon'}
    @torch.no_grad()
    def wrapped(*args):
        before = tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in args)
        original(*args)
        if not bool(torch.isfinite(args[1]).all()):
            bad = (~torch.isfinite(args[1])).flatten(1).any(1).nonzero().flatten().tolist()
            group_names = groups[tuple(args[1].shape[1:])]
            event = dict(step=current_step, shape=list(args[1].shape),
                bad_parameters=[group_names[i] for i in bad],
                inputs={k: stats(a) for k, a in zip(('gradients','parameters','momentum','second_moment'), before[:4])},
                after={k: stats(a) for k, a in zip(('gradients','parameters','momentum','second_moment'), args[:4])},
                eager_shadow=shadow(before), max_scaled_shadow=shadow(before, True))
            events.append(event)
            write(out/'optimizer-events.json', events)
    optim.muon_step_fused = wrapped
    rows = []; reason = 'budget_exhausted'
    try:
        for step in range(start, min(start+128, 640)):
            current_step = step+1
            schedule(opt, step, 305176, wd); opt.zero_grad(set_to_none=True)
            x, y = data.batch('train', step*16, 16)
            loss = model(x, y); loss.backward()
            gradients = {n: stats(p.grad) for n, p in model.named_parameters() if p.grad is not None}
            row = dict(step=current_step, loss=float(loss.detach()),
                max_gradient_abs=max(s['maxabs'] for s in gradients.values()),
                max_gradient_rms=max(s['rms'] for s in gradients.values()),
                all_gradients_finite=all(s['finite'] for s in gradients.values()))
            if not row['all_gradients_finite'] or not bool(loss.isfinite()):
                reason = 'nonfinite_before_optimizer'; rows.append(row)
                write(out/'failure-gradients.json', gradients); break
            opt.step()
            bad_params = {n: stats(p) for n, p in model.named_parameters() if not bool(p.isfinite().all())}
            row['bad_parameters_after_optimizer'] = bad_params
            rows.append(row)
            if bad_params:
                reason = 'first_nonfinite_parameters_after_optimizer'
                write(out/'failure-gradients.json', gradients); break
            if current_step%10==0:
                print('optimizer probe', current_step, row['loss'], row['max_gradient_abs'], flush=True)
        write(out/'replay.json', rows)
        summary = dict(complete=True, formal_training=False, weights_returned=False,
            reason=reason, last_step=current_step, steps=len(rows), events=len(events), gpu=torch.cuda.get_device_name())
        write(out/'summary.json', summary); print(json.dumps(summary), flush=True)
    finally:
        optim.muon_step_fused = original


if __name__ == '__main__':
    main()
