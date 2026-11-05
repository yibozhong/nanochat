"""Bounded failure diagnosis; never writes weights or starts formal training."""
import argparse
import gc
import json
import math
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from nanochat import flash_attention
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.connection_experiment import configure_attention
from nanochat.packed_training import PackedData, seed_all, restore_rng


def clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def write(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')


def stats(x):
    if x is None:
        return None
    x = x.detach().double()
    return dict(rms=float(x.square().mean().sqrt()), maxabs=float(x.abs().max()),
                finite=bool(x.isfinite().all()), nonfinite=int((~x.isfinite()).sum()))


@torch.no_grad()
def routing(module, x, u):
    b, t, d = x.shape
    q = module.q(u).view(b, t, module.heads, module.head_dim)
    k = module.k(x).view_as(q)
    v = module.v(x).view_as(q)
    result = dict(q=stats(q), k=stats(k), v=stats(v))
    if not all(result[n]['finite'] for n in ('q', 'k', 'v')):
        return result
    pos = torch.unique(torch.linspace(0, t-1, 32, device=x.device).long())
    offsets = torch.arange(module.window, device=x.device)
    keys = pos[:, None] - offsets
    mask = keys >= 0
    # [B, selected query, H, key, head_dim], including diagonal at key 0.
    ks = k[:, keys.clamp_min(0)].permute(0, 1, 3, 2, 4).double()
    vs = v[:, keys.clamp_min(0)].permute(0, 1, 3, 2, 4).double()
    scores = (q[:, pos].double().unsqueeze(-2)*ks).sum(-1)/math.sqrt(module.head_dim)
    visible = mask[None, :, None, :].expand_as(scores)
    result['logits'] = stats(scores[visible])
    a = scores.masked_fill(~visible, -float('inf')).softmax(-1)
    result.update(entropy_mean=float(-(a*a.clamp_min(1e-300).log()).sum(-1).mean()),
                  effective_support_mean=float(a.square().sum(-1).reciprocal().mean()),
                  diagonal_probability_mean=float(a[..., 0].mean()),
                  top1_probability_mean=float(a.max(-1).values.mean()))
    ref = (a.unsqueeze(-1)*vs).sum(-2)
    actual = flash_attention.flash_attn_func(q, k, v, causal=True,
        window_size=(module.window-1, 0))[:, pos].double()
    result['sampled_fa3_vs_fp64_relative_l2'] = float((actual-ref).norm()/ref.norm().clamp_min(1e-300))
    return result


def trace(model, x, y, attention=True, backward=True):
    saved = model.compiled_steps
    model.compiled_steps = None
    records, handles = [], []
    model.zero_grad(set_to_none=True)
    def hook(i):
        def capture(module, args, output):
            if backward:
                output.retain_grad()
            records.append((i, module, args[0], args[1], output))
        return capture
    for i, block in enumerate(model.blocks):
        handles.append(block.token_residual.register_forward_hook(hook(i)))
    try:
        with torch.set_grad_enabled(backward):
            loss = model(x, y)
            if backward:
                loss.backward()
        layers = []
        for j, (i, module, h, u, out) in enumerate(records):
            f = out.detach().double()
            power = f.square().mean()
            route = out.detach().double()-u.detach().double()
            layers.append(dict(layer=i, sublayer='attention' if j%2 == 0 else 'mlp',
                input=stats(h), branch=stats(u), output=stats(out), route=stats(route),
                gain=float(f.norm()/h.detach().double().norm().clamp_min(1e-300)),
                route_to_branch=float(route.norm()/u.detach().double().norm().clamp_min(1e-300)),
                token_variance_fraction=float(f.var(1, unbiased=False).mean()/power.clamp_min(1e-300)),
                output_gradient=stats(out.grad) if backward else None,
                attention=routing(module, h.detach(), u.detach()) if attention else None))
        return dict(ce=float(loss.detach()), layers=layers,
            parameter_gradients={n: stats(p.grad) for n, p in model.named_parameters()} if backward else {})
    finally:
        for handle in handles:
            handle.remove()
        model.compiled_steps = saved
        model.zero_grad(set_to_none=True)


@torch.no_grad()
def matrix_metrics(model):
    result = []
    product = torch.eye(model.config.width, device='cuda', dtype=torch.float64)
    logscale = 0.
    for i, block in enumerate(model.blocks):
        module = block.token_residual
        o, v = module.o.weight.double(), module.v.weight.double()
        c = o@v
        twice = c@c
        product = twice@product
        scale = product.norm().clamp_min(1e-300)
        logscale += float(scale.log())
        product /= scale
        result.append(dict(layer=i, o=stats(o), v=stats(v),
            o_smax=float(torch.linalg.svdvals(o)[0]), v_smax=float(torch.linalg.svdvals(v)[0]),
            ov_smax=float(torch.linalg.svdvals(c)[0]), ov_squared_smax=float(torch.linalg.svdvals(twice)[0]),
            constant_token_route_product_log_smax=logscale+float(torch.linalg.svdvals(product)[0].log())))
    return result


def intervention(model, mode):
    originals = []
    for block in model.blocks:
        module = block.token_residual
        originals.append(module.forward)
        def forward(self, x, u):
            b, t, d = x.shape
            # Diagnostic-only counterfactual; no parameter mutation.
            xn = F.rms_norm(x, (d,))
            kinput = xn if mode == 'kv_norm' else x
            vinput = xn if mode in ('v_norm', 'kv_norm') else x
            q = self.q(u).view(b, t, self.heads, self.head_dim)
            k = self.k(kinput).view_as(q)
            v = self.v(vinput).view_as(q)
            if mode == 'qk_norm':
                q, k = F.rms_norm(q, (self.head_dim,)), F.rms_norm(k, (self.head_dim,))
            r = flash_attention.flash_attn_func(q, k, v, causal=True, window_size=(self.window-1, 0))
            route = self.o(r.reshape(b, t, d))
            return u + route*(0.1 if mode == 'route_x0.1' else 1.)
        module.forward = MethodType(forward, module)
    return originals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data')
    parser.add_argument('--checkpoint-dir', default='../checkpoints')
    parser.add_argument('--output-dir', default='results/diagnostic')
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    configure_attention('fa3')
    assert flash_attention.USE_FA3
    data = PackedData(args.data_dir)
    x, y = data.batch('dev', 0, 2)
    config = ConnectionConfig(tokenres=True)
    for label in ('initial', 'best', 'latest'):
        seed_all(42)
        model = ConnectionModel(config).prepare()
        metadata = dict(step=0)
        if label != 'initial':
            state = torch.load(Path(args.checkpoint_dir)/(label+'.pt'), map_location='cpu', weights_only=False)
            assert ConnectionConfig.from_checkpoint(state['config']) == config
            model.load_state_dict(state['model'])
            metadata = {k: state[k] for k in ('step', 'run_id', 'source')}
            del state
        result = dict(label=label, checkpoint=metadata,
            parameter_finite=all(bool(p.isfinite().all()) for p in model.parameters()),
            matrices=matrix_metrics(model), dev=trace(model, x, y))
        write(out/(label+'.json'), result)
        print(label, 'step', metadata['step'], 'CE', result['dev']['ce'],
              'max RMS', max(r['output']['rms'] for r in result['dev']['layers']), flush=True)
        if label == 'latest':
            probes = {}
            for mode in ('v_norm', 'kv_norm', 'qk_norm', 'route_x0.1'):
                old = intervention(model, mode)
                try:
                    probes[mode] = trace(model, x, y, attention=False)
                finally:
                    for block, original in zip(model.blocks, old):
                        block.token_residual.forward = original
            write(out/'frozen-counterfactuals.json', probes)
        del model
        gc.collect(); torch.cuda.empty_cache()

    # Same immutable checkpoint, original optimizer/RNG/data/schedule. At most
    # 128 additional diagnostic steps; no new checkpoint or W&B run is created.
    state = torch.load(Path(args.checkpoint_dir)/'latest.pt', map_location='cpu', weights_only=False)
    seed_all(42)
    model = ConnectionModel(config).prepare(compile_blocks=True)
    model.load_state_dict(state['model']); opt, wd = optimizer_for(model)
    opt.load_state_dict(state['optimizer']); restore_rng(state['rng'])
    start = state['step']; del state
    rows = []; failure = None
    for step in range(start, min(start+128, 640)):
        schedule(opt, step, 305176, wd)
        bx, by = data.batch('train', step*16, 16)
        opt.zero_grad(set_to_none=True)
        raw_loss = model(bx, by)
        raw_loss.backward()
        try:
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            finite = torch.stack([torch.isfinite(g).all() for g in gradients]).all()
            if not bool(finite) or not bool(torch.isfinite(raw_loss)):
                raise FloatingPointError('Nonfinite loss/gradient before optimizer update')
            opt.step()
            loss = float(raw_loss.detach())
        except FloatingPointError as exc:
            failure = dict(attempted_step=step+1, message=str(exc),
                compiled_loss=float(raw_loss.detach()),
                compiled_parameter_gradients={n: stats(p.grad) for n, p in model.named_parameters()})
            write(out/'failure-compiled.json', failure)
            write(out/'failure-eager.json', trace(model, bx, by))
            break
        rows.append(dict(step=step+1, loss=loss))
        if (step+1) % 10 == 0:
            print('replay', step+1, loss, flush=True)
            write(out/'replay.json', dict(start_step=start, steps=rows, complete=False))
    write(out/'replay.json', dict(start_step=start, steps=rows, complete=True, failure=failure,
        note='Bounded diagnosis, not continued formal training; exact BF16 trajectory can differ across GPUs/processes.'))
    write(out/'summary.json', dict(complete=True, formal_training=False, weights_returned=False,
        gpu=torch.cuda.get_device_name(), original_failed_step=552, latest_step=start,
        replay_steps=len(rows), replay_failed_step=None if failure is None else failure['attempted_step']))


if __name__ == '__main__':
    main()
