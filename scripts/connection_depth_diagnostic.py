"""Read-only, routing-preserving depth diagnostics for the retained models.

No production forward/weights are modified. See research/connections/depth/README.md.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from nanochat.connection_model import ConnectionConfig, build_connection_model
from nanochat.connection_experiment import configure_attention
from nanochat.gpt import norm
from nanochat.nvidia_connections import expand_streams
from nanochat.packed_training import PackedData, evaluate, seed_all


def vectors(x):
    # Do NOT collapse mHC streams: [B,T,D,K] -> [B,T,D*K].
    return x.float().flatten(2)


def cosine(a, b):
    a, b = vectors(a), vectors(b)
    den = a.norm(dim=-1) * b.norm(dim=-1)
    return torch.where(den > 1e-20, (a*b).sum(-1)/den.clamp_min(1e-20), torch.nan)


def measurements(before, after, compute, routing):
    h, total, c, r = map(vectors, (before, after.float()-before.float(), compute, routing))
    hn = h.norm(dim=-1)
    assert bool((hn > 1e-12).all()), 'Undefined relative update: zero residual'
    ans = dict(total_ratio=total.norm(dim=-1)/hn, compute_ratio=c.norm(dim=-1)/hn,
               routing_ratio=r.norm(dim=-1)/hn, hidden_norm=hn,
               hidden_variance=h.var(dim=-1, correction=0), hidden_energy=h.square().sum(-1),
               total_energy=total.square().sum(-1), compute_energy=c.square().sum(-1),
               routing_energy=r.square().sum(-1), hidden_cosine=cosine(before, after),
               update_residual_cosine=cosine(total, h))
    err = (total-c-r).abs().max().item()
    assert err <= 1e-5, ('decomposition', err)
    return ans, err


def sublayer(x, connection, branch, disabled, observe):
    if connection is None:
        y = torch.zeros_like(x) if disabled else branch(x)
        after = x + y
        pure = x
    else:
        h, residual, mixes = connection.read(x)
        y = torch.zeros_like(h) if disabled else branch(h)
        after = connection.write(y, residual, mixes)
        # Reuse THIS invocation's coefficients: not a second routing estimate.
        pure = connection.write(torch.zeros_like(y), residual, mixes) if observe else None
    if not observe:
        return after, None
    compute = after.float()-pure.float()
    routing = pure.float()-x.float()
    stats, err = measurements(x, after, compute, routing)
    # Rounding-inclusive increments telescope exactly. Also expose the
    # unrounded H_post * branch for interpretation, separately from routing.
    analytic = y.float() if connection is None else y.float().unsqueeze(-1)*mixes[0].float().unsqueeze(-2)
    stats['branch_rounding_ratio'] = vectors(compute-analytic).norm(dim=-1)/vectors(x).norm(dim=-1)
    return after, (compute, routing, stats, err)


@torch.no_grad()
def traced_hidden(model, idx, skip=None, observe=False):
    """Evaluation-only expansion of the existing hidden() with the same ops.

    A skipped block disables both attention and MLP, keeping native affine
    routing and dynamic mHC routing (recomputed along the intervened path).
    """
    cfg = model.config
    native = cfg.architecture == 'native'
    core = model.native if native else model
    blocks = core.transformer.h if native else core.blocks
    embed = core.transformer.wte if native else core.wte
    x = norm(embed(idx).to(torch.bfloat16))
    if native:
        gate = core.smear_lambda.to(x.dtype)*torch.sigmoid(core.smear_gate(x[:, 1:, :24]))
        x = torch.cat([x[:, :1], x[:, 1:]+gate*x[:, :-1]], dim=1)
    if cfg.variant != 'baseline':
        x = expand_streams(x, cfg.streams)
    x0, backout = x, None
    cos_sin = core.cos[:, :idx.shape[1]], core.sin[:, :idx.shape[1]]
    metrics, submetrics, updates, branch_updates, errors = [], [], [], [], []
    for i, block in enumerate(blocks):
        before = x
        if native:
            x = core.resid_lambdas[i]*x + core.x0_lambdas[i]*x0
            ve = core.value_embeds[str(i)](idx).to(x.dtype) if str(i) in core.value_embeds else None
            window = core.window_sizes[i]
        else:
            ve, window = None, (-1, 0)
        affine = x.float()-before.float() if observe else None
        ac = block.attn_connection if cfg.variant != 'baseline' else None
        mc = block.mlp_connection if cfg.variant != 'baseline' else None
        x, a = sublayer(x, ac, lambda h: block.attn(norm(h), ve, cos_sin, window, None), i == skip, observe)
        x, m = sublayer(x, mc, lambda h: block.mlp(norm(h)), i == skip, observe)
        if observe:
            comp, route = a[0]+m[0], affine+a[1]+m[1]
            row, err = measurements(before, x, comp, route)
            row['native_affine_ratio'] = vectors(affine).norm(dim=-1)/vectors(before).norm(dim=-1)
            metrics.append(row); submetrics.extend([a[2], m[2]])
            updates.append(x.float()-before.float()); branch_updates.append(comp)
            errors.extend([a[3], m[3], err])
        if native and i == cfg.depth//2:
            backout = x
    final_extra = None
    if native and backout is not None:
        before = x
        x = x-core.backout_lambda.to(x.dtype)*backout
        if observe:
            final_extra = (vectors(x.float()-before.float()).norm(dim=-1)/vectors(before).norm(dim=-1)).mean(1).cpu().tolist()
    if core.stream_head is not None:
        x = core.stream_head(x)
    if not observe:
        return norm(x), None
    # Each sequence is one statistical unit, rather than 2048 independent tokens.
    pack = lambda rows: {k: torch.stack([r[k].mean(1) for r in rows], dim=1).cpu().numpy() for k in rows[0]}
    adjacent = {}
    for name, us in [('total', updates), ('compute', branch_updates)]:
        cs = torch.stack([cosine(us[i], us[i+1]) for i in range(len(us)-1)], dim=-1)
        adjacent[name] = torch.nanmean(cs, dim=1).cpu().numpy()
        adjacent[name+'_valid_tokens'] = torch.isfinite(cs).sum(1).cpu().numpy()
    return norm(x), dict(layer=pack(metrics), sublayer=pack(submetrics), adjacent=adjacent,
                        decomposition_max_abs=max(errors), native_backout_ratio=final_extra)


def per_sequence_ce(model, h, targets):
    from cut_cross_entropy import linear_cross_entropy
    head = model.native.lm_head if model.config.architecture == 'native' else model.lm_head
    weight = head.weight[:model.config.vocab_size].to(h.dtype)
    ce = linear_cross_entropy(h, weight, targets, softcap=15., shift=0, reduction='none',
        filter_eps=None, filter_e_grad=False, filter_c_grad=False, accum_e_fp32=True, accum_c_fp32=True)
    assert ce.shape == targets.shape
    return ce.float().mean(1)


@contextmanager
def zero_branch_oracle(model, layer):
    """Independent reference: original forward, zero module outputs via hooks."""
    blocks = model.native.transformer.h if model.config.architecture == 'native' else model.blocks
    handles = [module.register_forward_hook(lambda module, args, output: torch.zeros_like(output))
               for module in (blocks[layer].attn, blocks[layer].mlp)]
    try:
        yield
    finally:
        for h in handles: h.remove()


def fingerprint(model):
    digest = hashlib.sha256()
    # Include nonpersistent buffers (e.g. RoPE lookup tables) as well as weights.
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    for name, tensor in tensors:
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def summarized(values):
    a = np.asarray(values, dtype=np.float64)
    assert np.isfinite(a).all()
    return dict(mean=float(a.mean()), sequence_se=float(a.std(ddof=1)/np.sqrt(len(a))),
                ci95=[float(x) for x in np.quantile(a[np.random.default_rng(42).integers(0,len(a),(4000,len(a)))].mean(1),[.025,.975])])


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True); p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', default='results/depth'); p.add_argument('--microbatch', type=int, default=4)
    a = p.parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    configure_attention('sdpa_cudnn'); seed_all(42)
    data = PackedData(a.data_dir); assert data.manifest['sequence_length'] == 2048
    specs = json.loads(Path(a.inputs).read_text())
    results = dict(sequence_length=2048, split='dev', rows=len(data.arrays['dev']),
        validation_tokens=len(data.arrays['dev'])*2048, gpu=torch.cuda.get_device_name(),
        attention_backend='sdpa_cudnn', formal_training=False, logging_backend='local', models={})
    started = time.time()
    for name, spec in specs.items():
        print(json.dumps(dict(model=name,phase='load_checkpoint_and_verify')),flush=True)
        state = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
        cfg = ConnectionConfig.from_checkpoint(state['config'])
        assert cfg.depth == 8 and cfg.sequence_length == 2048
        assert state['step'] == 305176 and state['protocol']['seed'] == 42
        assert state['data_manifest']['files']['dev']['sha256'] == data.manifest['files']['dev']['sha256']
        model = build_connection_model(cfg).prepare(compile_blocks=False).eval()
        model.load_state_dict(state['model'], strict=True); model.requires_grad_(False)
        meta = dict(config=asdict(cfg), parameters=sum(p.numel() for p in model.parameters()),
                    checkpoint_step=state['step'], training_tokens=state['step']*32768,
                    run_id=state['run_id'], historical_dev_ce=spec['historical_dev_ce'],
                    training_protocol=state['protocol'], dev_sha256=state['data_manifest']['files']['dev']['sha256'])
        del state; gc.collect()
        initial_hash = fingerprint(model)
        x, y = data.batch('dev',0,2)
        expected = model.hidden(x)
        got, _ = traced_hidden(model,x)
        torch.testing.assert_close(got, expected, atol=0, rtol=0)
        skip_errors = []
        for layer in range(cfg.depth):
            got, _ = traced_hidden(model,x,skip=layer)
            with zero_branch_oracle(model,layer):
                expected = model.hidden(x)
            error = (got.float()-expected.float()).abs().max().item(); skip_errors.append(error)
            torch.testing.assert_close(got,expected,atol=0,rtol=0)
        # Confirm the diagnostic actually bypasses compute, not merely its write.
        blocks = model.native.transformer.h if cfg.architecture == 'native' else model.blocks
        calls = []
        handles = [module.register_forward_pre_hook(lambda *args: calls.append(True)) for module in (blocks[3].attn, blocks[3].mlp)]
        traced_hidden(model,x,skip=3)
        for h in handles: h.remove()
        assert not calls
        ordinary_ce, count = evaluate(model,data,microbatch=a.microbatch)
        baseline, skipped, traces = [], [], []
        max_error = 0.
        for start in range(0,len(data.arrays['dev']),a.microbatch):
            x,y = data.batch('dev',start,min(a.microbatch,len(data.arrays['dev'])-start))
            h, trace = traced_hidden(model,x,observe=True)
            baseline.append(per_sequence_ce(model,h,y).cpu().numpy()); traces.append(trace)
            max_error = max(max_error,trace['decomposition_max_abs'])
            losses = []
            for layer in range(cfg.depth):
                h,_ = traced_hidden(model,x,skip=layer)
                losses.append(per_sequence_ce(model,h,y).cpu().numpy())
            skipped.append(np.stack(losses,axis=1))
        baseline = np.concatenate(baseline); skipped = np.concatenate(skipped); delta = skipped-baseline[:,None]
        assert abs(float(baseline.mean())-ordinary_ce) < 2e-6
        assert abs(ordinary_ce-spec['historical_dev_ce']) < .002, 'Unexpected checkpoint/evaluation discrepancy'
        raw = dict(baseline_ce=baseline,skip_ce=skipped,delta_ce=delta)
        for category in ('layer','sublayer','adjacent'):
            for key in traces[0][category]:
                raw[category+'_'+key] = np.concatenate([t[category][key] for t in traces])
        if cfg.architecture == 'native':
            raw['native_backout_ratio'] = np.concatenate([t['native_backout_ratio'] for t in traces])
        np.savez_compressed(out/(name+'-per-sequence.npz'),**raw)
        meta.update(baseline_ce=summarized(baseline), ordinary_evaluate_ce=ordinary_ce,
            historical_ce_difference=ordinary_ce-spec['historical_dev_ce'],
            checks=dict(unmodified_forward_exact=True, skip_oracle_max_abs=skip_errors,
                        skipped_branches_not_called=True, decomposition_max_abs=max_error,
                        weights_and_buffers_unchanged=initial_hash==fingerprint(model)),layers=[],sublayers=[])
        assert meta['checks']['weights_and_buffers_unchanged']
        for l in range(cfg.depth):
            row=dict(layer=l+1,normalized_depth=(l+1)/cfg.depth,
                     skip_ce=summarized(skipped[:,l]),delta_ce=summarized(delta[:,l]),
                     relative_ce_increase=float(delta[:,l].mean()/baseline.mean()))
            for k in traces[0]['layer']:
                row[k] = summarized(raw['layer_'+k][:,l])
            for kind in ('total','compute','routing'):
                row[kind+'_frobenius_ratio'] = float(np.sqrt(raw['layer_'+kind+'_energy'][:,l].sum()/raw['layer_hidden_energy'][:,l].sum()))
            if l < cfg.depth-1:
                for kind in ('total','compute'):
                    row['adjacent_'+kind+'_cosine'] = summarized(raw['adjacent_'+kind][:,l])
                    row['adjacent_'+kind+'_valid_tokens'] = int(raw['adjacent_'+kind+'_valid_tokens'][:,l].sum())
            meta['layers'].append(row)
        for j in range(2*cfg.depth):
            row=dict(layer=j//2+1,branch='attention' if j%2==0 else 'mlp',normalized_depth=(j//2+1)/cfg.depth)
            for k in traces[0]['sublayer']:row[k]=summarized(raw['sublayer_'+k][:,j])
            meta['sublayers'].append(row)
        meta['early_mean_delta_ce'] = summarized(delta[:,:4].mean(1))
        meta['late_mean_delta_ce'] = summarized(delta[:,4:].mean(1))
        meta['late_minus_early_delta_ce'] = summarized(delta[:,4:].mean(1)-delta[:,:4].mean(1))
        meta['late_except_final_mean_delta_ce'] = summarized(delta[:,4:7].mean(1))
        meta['positive_importance_late_share'] = float(np.maximum(delta.mean(0),0)[4:].sum()/np.maximum(delta.mean(0),0).sum())
        if cfg.architecture=='native':meta['native_backout_ratio']=summarized(raw['native_backout_ratio'])
        results['models'][name] = meta
        (out/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(model=name,baseline_ce=ordinary_ce,layer_delta_ce=delta.mean(0).tolist(),checks=meta['checks'])),flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    results.update(complete=True,seconds=time.time()-started)
    (out/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
