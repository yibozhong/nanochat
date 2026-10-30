"""Audit the returned archive; no model execution or GPU work."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def main():
    root = Path(sys.argv[1]).resolve()
    state = json.loads((root / 'state.json').read_text())
    archive = root / 'results' / f'gate-{state["job"]}.tar'
    result = json.loads((root / 'comparison.json').read_text())
    assert result['complete'] and result['wandb_verified']
    trials = result['trials']
    assert len({v['config']['gpu_uuid'] for v in trials.values()}) == 1
    rows = {}
    base_ms = trials['baseline']['normal_ms']['mean']
    old_ms = trials['mhc_official']['normal_ms']['mean']
    compiled_ms = trials['mhc_torch_compile']['normal_ms']['mean']
    for name, v in trials.items():
        kernels = v['cuda_profile']['kernels']
        def matching(fragment):
            return [k for k in kernels if fragment in k['name']]
        old = matching('_mhc_pre_norm_fn')
        nv_fwd = matching('_mhc_projection_fwd_fused')
        nv_bwd = matching('_mhc_projection_bwd_fused_dx')
        if name.startswith('mhc_nvidia'):
            assert not old and nv_fwd and nv_bwd, (name, old, nv_fwd, nv_bwd)
            # Two instrumented steps, 16 branches plus final stream head each.
            assert sum(k['count'] for k in nv_fwd) == 34
            assert sum(k['count'] for k in nv_bwd) == 34
        else:
            assert not nv_fwd and not nv_bwd
        ms = v['normal_ms']['mean']
        rows[name] = dict(mean_ms=ms, stdev_ms=v['normal_ms']['stdev'],
            median_ms=v['normal_ms']['median'],max_ms=max(v['normal_ms']['samples']),
            overhead_vs_baseline_pct=100*(ms/base_ms-1),
            reduction_vs_official_pct=100*(1-ms/old_ms),
            reduction_vs_compiled_pct=100*(1-ms/compiled_ms),
            memory_gib={k: b/1024**3 for k,b in v['normal_timing_memory'].items()},
            phase_means=v['phase_means'], run_id=v['run_id'],
            projection_kernels=old+nv_fwd+nv_bwd)
    with tarfile.open(archive) as tar:
        errors = {p.name: json.load(tar.extractfile(p)) for p in tar.getmembers()
                  if p.name.startswith('results/torch-projection-checks/') and p.name.endswith('.json')}
        assert len(errors) == 8, errors.keys()
        separate = {p.name: json.load(tar.extractfile(p)) for p in tar.getmembers()
                    if p.isfile() and p.name.startswith('results/oracle-') and p.name.endswith('.json')}
    audit = dict(job=state['job'], source=result['source'], checks=result['checks'],
        gpu=trials['baseline']['config']['gpu'], gpu_uuid=trials['baseline']['config']['gpu_uuid'],
        same_gpu=True, upstream_kernel_trace_verified=True, numerical_errors=separate or errors,
        timings=rows, microbenchmark=result['microbenchmark'],
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        limitations=['Short diagnostics only, no long-run quality result',
                    'Explicit precision variants are not bitwise-equivalent',
                    'Timing includes ordinary gradient-finiteness checking',
                    'Memory is PyTorch allocator peak, not whole-device use'])
    destination = Path(sys.argv[2]) if len(sys.argv)>2 else Path(__file__).parent
    (destination / 'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(job=state['job'],gpu=audit['gpu'],checks=audit['checks'],
        timings={k:{a:b for a,b in v.items() if a not in ('projection_kernels','phase_means')}
                 for k,v in rows.items()}),indent=2))


if __name__ == '__main__':
    main()
