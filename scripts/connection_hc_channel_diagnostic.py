"""Frozen 10B no-Sinkhorn HC diagnostics; no optimizer or checkpoint writes.

H is indexed [input stream, output stream]. Geometry is observational: the
production write runs unchanged; only small sampled tensors are decomposed.
Scalar interventions preserve calibration input energy, not output geometry.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model
from nanochat.packed_training import PackedData, json_write, seed_all
from scripts.connection_depth_diagnostic import fingerprint, per_sequence_ce, summarized


def sites(model):
    return [(f'{i+1:02d}_{kind}', getattr(block, kind+'_connection'))
            for i, block in enumerate(model.blocks) for kind in ('attn', 'mlp')]


def ratio(a, b):
    return torch.where(b > 1e-20, a / b.clamp_min(1e-20), float('nan'))


def geometry(x):
    """Per-token stream geometry, excluding undefined zero-vector statistics."""
    x = x.double()
    gram = x.transpose(-1, -2) @ x
    energy = gram.diagonal(dim1=-2, dim2=-1).sum(-1)
    difference = (x-x.mean(-1, keepdim=True)).square().sum((-2, -1))
    eigen = torch.linalg.eigvalsh(gram).clamp_min(0)
    prob = ratio(eigen, energy[..., None])
    erank = (-(prob*prob.clamp_min(1e-30).log()).sum(-1)).exp()
    erank = torch.where(energy > 1e-20, erank, float('nan'))
    diag = gram.diagonal(dim1=-2, dim2=-1)
    ij = torch.triu_indices(x.shape[-1], x.shape[-1], offset=1, device=x.device)
    cos = ratio(gram[..., ij[0], ij[1]], (diag[..., ij[0]]*diag[..., ij[1]]).sqrt())
    return dict(energy=energy, difference_energy=difference,
                difference_fraction=ratio(difference, energy), effective_rank=erank,
                stable_rank=ratio(energy, eigen[..., -1]),
                mean_pair_cosine=torch.nanmean(cos, -1),
                valid_pair_cosines=torch.isfinite(cos).sum(-1).double())


def scalar_replacements(scale, x2):
    """Per-stream constants fitted to original train-calibration activations.

    x2[c,i] = sum_tokens x[token,c,i]^2. Energy control exactly preserves
    sum(x*scale)^2 on these fixed inputs, while removing all channel variation.
    LS control minimizes sum x^2*(scale-a)^2. Neither changes weights permanently.
    """
    s, w = scale.double(), x2.double()
    den = w.sum(0)
    assert bool((den > 0).all())
    ls = (w*s).sum(0)/den
    rms = ((w*s.square()).sum(0)/den).sqrt()
    rms = rms*torch.where(ls < 0, -1., 1.)
    return {k: v[None, :].expand_as(scale).to(scale.dtype).contiguous()
            for k, v in dict(channel_mean=s.mean(0), calibrated_ls=ls,
                             calibrated_energy=rms).items()}


@contextmanager
def replace_scales(model, replacements):
    modules = dict(sites(model))
    saved = {k: modules[k].input_scale.detach().clone() for k in replacements}
    try:
        with torch.no_grad():
            for k, value in replacements.items():
                assert value.shape == modules[k].input_scale.shape
                modules[k].input_scale.copy_(value)
        yield
    finally:
        with torch.no_grad():
            for k, value in saved.items(): modules[k].input_scale.copy_(value)


class Observer:
    def __init__(self, model, mode, stride=32):
        self.model, self.mode, self.stride = model, mode, stride
        self.rows, self.x2, self.decomposition_relative = {}, {}, {}

    def observe(self, name, module, branch, residual, mixes, actual):
        if self.mode == 'calibrate':
            energy = residual.float().square().sum((0, 1), dtype=torch.float64)
            self.x2[name] = self.x2.get(name, torch.zeros_like(energy)) + energy
            return
        post, h, buffer = mixes
        assert buffer is None and not module.sinkhorn
        x = residual[:, ::self.stride].double()
        y = branch[:, ::self.stride].double()
        h = h[:, ::self.stride].double()
        post = post[:, ::self.stride].double()
        scale = getattr(module, 'input_scale', torch.ones_like(x[0, 0])).double()
        z = x*scale
        mixed = z @ h
        write = y[..., None]*post[..., None, :]
        after = actual[:, ::self.stride].double()
        rel = ((after-mixed-write).square().sum()/after.square().sum().clamp_min(1e-20)).sqrt().item()
        self.decomposition_relative[name] = max(rel, self.decomposition_relative.get(name, 0.))
        assert rel < .01, (name, rel, 'Incorrect mix orientation/decomposition')
        stats = {}
        stages = {k: geometry(v) for k, v in dict(input=x, pregated=z, mixed=mixed,
                                                  branch_write=write, output=after).items()}
        for stage, values in stages.items():
            stats.update({stage+'_'+k: v for k, v in values.items()})
        for before in ('input', 'pregated'):
            stats['mix_over_'+before+'_difference_energy'] = ratio(
                stages['mixed']['difference_energy'], stages[before]['difference_energy'])
        singular = torch.linalg.svdvals(h)
        stats.update(h_sigma_max=singular[..., 0], h_sigma_min=singular[..., -1],
                     h_negative_fraction=(h < 0).double().mean((-2, -1)),
                     h_row_sum_error=(h.sum(-1)-1).abs().mean(-1),
                     h_column_sum_error=(h.sum(-2)-1).abs().mean(-1))
        record = {k: torch.nanmean(v, dim=1).cpu().numpy() for k, v in stats.items()}
        record.update({k+'_valid_tokens': torch.isfinite(v).sum(1).cpu().numpy()
                       for k, v in stats.items()})
        self.rows.setdefault(name, []).append(record)

    @contextmanager
    def installed(self):
        saved = []
        try:
            for name, module in sites(self.model):
                original = module.write
                had_instance = 'write' in module.__dict__
                def write(branch, residual, mixes, *, _name=name, _module=module, _original=original):
                    actual = _original(branch, residual, mixes)
                    self.observe(_name, _module, branch, residual, mixes, actual)
                    return actual
                saved.append((module, original, had_instance))
                module.write = write
            yield self
        finally:
            for module, original, had_instance in saved:
                if had_instance: module.write = original
                else: del module.write


@torch.no_grad()
def losses(model, data, microbatch, observer=None, split='dev', rows=None):
    result = []
    limit = len(data.arrays[split]) if rows is None else rows
    for start in range(0, limit, microbatch):
        x, y = data.batch(split, start, min(microbatch, limit-start))
        h = model.hidden(x)
        if split == 'dev': result.append(per_sequence_ce(model, h, y).cpu().numpy())
    if split != 'dev': return None
    result = np.concatenate(result)
    assert np.isfinite(result).all()
    return result


def gate_statistics(scale):
    s = scale.double()
    centered = s-s.mean(0, keepdim=True)
    return dict(min=s.min().item(), max=s.max().item(),
                mean_by_stream=s.mean(0).tolist(), std_by_stream=s.std(0).tolist(),
                channel_variance_energy_fraction=(centered.square().sum()/s.square().sum()).item())


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True); p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', default='results/hc-channel-diagnostic')
    p.add_argument('--microbatch', type=int, default=4)
    a = p.parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); seed_all(42)
    # Keep diagnostic Gram products precise; this flag does not alter TE's explicit TF32 policy.
    torch.backends.cuda.matmul.allow_tf32 = False
    backend = configure_attention('hardware_auto')
    data = PackedData(a.data_dir)
    assert data.manifest['sequence_length'] == 2048 and len(data.arrays['dev']) == 64
    specs = json.loads(Path(a.inputs).read_text())
    assert set(specs) == {'shared', 'scaled'}
    report = dict(complete=False, formal_training=False, optimizer_updates=0,
                  source=json.loads(Path('source.json').read_text()),
                  gpu=torch.cuda.get_device_name(), attention_backend=backend,
                  sequence_length=2048, dev_sequences=64, dev_tokens=131072,
                  calibration_split='train', calibration_rows=32, geometry_token_stride=32,
                  calibration_data_manifest=data.manifest,
                  models={}, limitations=['Single trained seed; frozen intervention is not a retrained scalar control.',
                      'Calibration scalar matches input energy on fixed calibration activations only.',
                      'Stream rank/cosine is descriptive, not a direct information or quality measure.',
                      'Mix-only gates do not alter the rank-one branch write formula.',
                      'Original training attention backends differ; re-evaluation uses one common backend.'])
    started = time.time(); original_losses = {}; manifests = {}
    for name, spec in specs.items():
        print(json.dumps(dict(model=name, phase='load')), flush=True)
        state = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
        cfg = ConnectionConfig.from_checkpoint(state['config'])
        assert cfg.architecture == 'vanilla' and cfg.variant == spec['variant']
        assert cfg.depth == 8 and cfg.width == 512 and cfg.sequence_length == 2048
        assert state['step'] == 305176 and state['protocol']['seed'] == 42
        assert state['run_id'] == spec['run_id']
        assert state['data_manifest']['files']['dev']['sha256'] == data.manifest['files']['dev']['sha256']
        manifests[name] = state['data_manifest']
        model = build_connection_model(cfg).prepare(compile_blocks=False).eval()
        model.load_state_dict(state['model'], strict=True); model.requires_grad_(False)
        assert all(not m.sinkhorn for _, m in sites(model))
        meta = dict(config=asdict(cfg), checkpoint_step=state['step'], run_id=state['run_id'],
                    checkpoint_sha256=spec['checkpoint_sha256'], training_protocol=state['protocol'],
                    training_data_manifest=state['data_manifest'], geometry={}, interventions={})
        del state; gc.collect()
        initial_hash = fingerprint(model)
        x, y = data.batch('dev', 0, a.microbatch)
        expected = model.hidden(x)
        parity = Observer(model, 'geometry')
        with parity.installed(): got = model.hidden(x)
        torch.testing.assert_close(got, expected, atol=0, rtol=0)
        del expected, got, parity
        observer = Observer(model, 'geometry')
        with observer.installed(): ce = losses(model, data, a.microbatch)
        assert abs(float(ce.mean())-spec['historical_dev_ce']) < .002, 'Historical CE mismatch'
        original_losses[name] = ce
        raw = dict(original_ce=ce)
        meta.update(original_ce=summarized(ce), historical_dev_ce=spec['historical_dev_ce'],
                    historical_ce_difference=float(ce.mean())-spec['historical_dev_ce'])
        for site, batches in observer.rows.items():
            meta['geometry'][site] = {}
            for key in batches[0]:
                values = np.concatenate([b[key] for b in batches])
                raw['geometry_'+site+'_'+key] = values
                meta['geometry'][site][key] = (int(values.sum()) if key.endswith('_valid_tokens')
                                                else summarized(values))
        meta['checks'] = dict(no_sinkhorn=True, observer_forward_exact=True,
                              decomposition_relative=observer.decomposition_relative)
        print(json.dumps(dict(model=name, phase='original', ce=float(ce.mean()))), flush=True)
        if name == 'scaled':
            calib = Observer(model, 'calibrate')
            with calib.installed(): losses(model, data, a.microbatch, split='train', rows=32)
            originals = {k: m.input_scale.detach().clone() for k, m in sites(model)}
            controls = {k: scalar_replacements(s, calib.x2[k]) for k, s in originals.items()}
            variants = {'ones': {k: torch.ones_like(s) for k, s in originals.items()}}
            for method in ('channel_mean', 'calibrated_ls', 'calibrated_energy'):
                variants[method] = {k: c[method] for k, c in controls.items()}
            # Permute channels jointly across the four streams: preserves every
            # per-stream distribution AND cross-stream gate-vector relationships.
            for seed in (42, 1337, 2026):
                gen = torch.Generator(device='cpu').manual_seed(seed)
                variants['shuffle_'+str(seed)] = {
                    k: s[torch.randperm(s.shape[0], generator=gen).to(s.device)].contiguous()
                    for k, s in originals.items()}
            for k in originals: variants['single_'+k] = {k: controls[k]['calibrated_energy']}
            meta['gate_statistics'] = {k: gate_statistics(s) for k, s in originals.items()}
            meta['calibration'] = {}
            for k, s in originals.items():
                w = calib.x2[k]; energy = (w*s.double().square()).sum(0)
                control = controls[k]['calibrated_energy']
                err = ((w*control.double().square()).sum(0)/energy-1).abs().max().item()
                assert err < 1e-6
                meta['calibration'][k] = dict(energy_relative_error=err,
                    energy_scalar=control[0].tolist(), ls_scalar=controls[k]['calibrated_ls'][0].tolist())
            for method, replacements in variants.items():
                with replace_scales(model, replacements): changed = losses(model, data, a.microbatch)
                delta = changed-ce
                raw[method+'_ce'], raw[method+'_delta_ce'] = changed, delta
                meta['interventions'][method] = dict(ce=summarized(changed), delta_ce=summarized(delta))
                print(json.dumps(dict(model=name, phase=method, ce=float(changed.mean()),
                                      delta_ce=float(delta.mean()))), flush=True)
                json_write(out/'progress.json', meta)
            assert len(meta['interventions']) == 23
        meta['checks']['weights_and_buffers_unchanged'] = initial_hash == fingerprint(model)
        assert meta['checks']['weights_and_buffers_unchanged']
        np.savez_compressed(out/(name+'-per-sequence.npz'), **raw)
        report['models'][name] = meta
        json_write(out/'partial.json', report)
        del model, observer; gc.collect(); torch.cuda.empty_cache()
    # Compare stored training file hashes across models, not the much smaller calibration pack.
    hashes = {k: {s: v['sha256'] for s, v in m['files'].items()} for k, m in manifests.items()}
    report['training_file_hashes_equal'] = hashes['shared'] == hashes['scaled']
    report['paired_scaled_minus_shared_ce'] = summarized(original_losses['scaled']-original_losses['shared'])
    report.update(complete=True, elapsed_seconds=time.time()-started)
    json_write(out/'diagnostic.json', report)
    print(json.dumps(dict(complete=True, paired_delta=report['paired_scaled_minus_shared_ce'],
                         training_file_hashes_equal=report['training_file_hashes_equal'])), flush=True)


if __name__ == '__main__': main()
