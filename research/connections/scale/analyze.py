"""Audit the same-GPU scaling experiment, including kernels and paired repeats."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def analyze(result):
    assert result['complete'] and (result.get('local_verified') or result.get('wandb_verified'))
    assert result['checks'] == dict(tests=9, failures=0, errors=0, skipped=0)
    trials = list(result['trials'].values())
    assert len(trials) == 12
    assert len({v['config']['gpu_uuid'] for v in trials}) == 1
    micro = result['common_microbatch']; accum = 16 // micro
    rows = []; details = []
    by_key = {}
    for v in trials:
        c = v['config']; model = c['model']
        key = (model['width'], model['variant'], c['repetition'])
        assert key not in by_key; by_key[key] = v
        assert v['source'] == result['source'] and v['optimizer_updates'] == 86
        assert v['training_tokens'] == 86 * 32768
        assert c['microbatch'] == micro and c['grad_accum'] == accum and c['batch_tokens'] == 32768
        assert c['seed'] == 42 and model['depth'] == 16 and model['heads'] * 128 == model['width']
        assert model['sequence_length'] == 2048 and model['vocab_size'] == 32768
        assert not model['activation_checkpoint'] and not model['fused_connection_grad']
        assert c['attention_backend'] == 'SDPA_CUDNN_ATTENTION_only'
        assert len(v['normal_ms']['samples']) == 60
        assert all(math.isfinite(x) and x > 0 for x in v['normal_ms']['samples'])
        assert math.isclose(statistics.mean(v['normal_ms']['samples']), v['normal_ms']['mean'], rel_tol=1e-12)
        kernels = v['cuda_profile']['kernels']; counts = {}
        attention = [k for k in kernels if any(x in k['name'].lower() for x in ('flash', 'sdpa', 'fmha', 'attention'))]
        assert attention and all('cudnn' in k['name'].lower() for k in attention)
        if model['variant'] == 'mhc':
            assert model['connection_backend'] == 'nvidia' and c['nvidia_tf32_override'] is None
            assert not any(any(x in k['name'] for x in ('mhc_pre_', 'mhc_post', 'mhc_head', 'expand_to_mhc')) for k in kernels)
            for fragment in ('_mhc_projection_fwd_fused', '_mhc_projection_bwd_fused_dx',
                             '_mhc_scale_fwd_fused', '_mhc_scale_bwd_fused',
                             '_mhc_aggregate_fwd', '_mhc_aggregate_bwd',
                             '_mhc_expand_combine_fwd', '_mhc_expand_combine_bwd',
                             '_mhc_sinkhorn_fwd_fused', '_mhc_sinkhorn_bwd_fused'):
                expected = (64 if 'expand_combine' in fragment or 'sinkhorn' in fragment else 66) * accum
                counts[fragment] = sum(k['count'] for k in kernels if fragment in k['name'])
                assert counts[fragment] == expected, (key, fragment, counts[fragment], expected)
        else:
            assert c['nvidia_tf32_override'] == '0'
            assert c['parameters'] == 12 * 16 * model['width']**2 + 2 * 32768 * model['width']
            assert not any('_mhc_' in k['name'] for k in kernels)
        phase = v['phase_means']['stream_elapsed_ms']
        details.append(dict(width=model['width'], variant=model['variant'], repetition=c['repetition'],
            run_id=v['run_id'], mean_ms=v['normal_ms']['mean'], median_ms=v['normal_ms']['median'],
            stdev_ms=v['normal_ms']['stdev'], forward_backward_event_ms=phase['forward']+phase['backward'],
            optimizer_event_ms=phase['optimizer'], finite_check_event_ms=phase['finite_check'],
            allocated_gib=v['normal_timing_memory']['peak_allocated_bytes']/1024**3,
            reserved_gib=v['normal_timing_memory']['peak_reserved_bytes']/1024**3,
            nvidia_kernel_counts=counts,
            attention_kernels=[dict(name=k['name'], ms_per_step=k['ms_per_step'], count=k['count']) for k in attention]))
    for width in (2048, 2560, 3072):
        variants = {}
        for variant in ('baseline', 'mhc'):
            vv = [by_key[(width, variant, repeat)] for repeat in (0, 1)]
            samples = [t for v in vv for t in v['normal_ms']['samples']]
            phases = [v['phase_means']['stream_elapsed_ms'] for v in vv]
            variants[variant] = dict(parameters=vv[0]['config']['parameters'],
                mean_ms=statistics.mean(samples), median_ms=statistics.median(samples),
                repeat_mean_ms=[v['normal_ms']['mean'] for v in vv],
                tokens_per_second=32768000/statistics.mean(samples),
                forward_backward_event_ms=statistics.mean(p['forward']+p['backward'] for p in phases),
                optimizer_event_ms=statistics.mean(p['optimizer'] for p in phases),
                reserved_gib=max(v['normal_timing_memory']['peak_reserved_bytes'] for v in vv)/1024**3,
                run_ids=[v['run_id'] for v in vv])
        base, mhc = variants['baseline'], variants['mhc']
        paired = [100*(by_key[(width, 'mhc', r)]['normal_ms']['mean']/
                       by_key[(width, 'baseline', r)]['normal_ms']['mean'] - 1) for r in (0, 1)]
        for repeat in (0, 1):
            pair = [next(d for d in details if (d['width'], d['variant'], d['repetition']) == (width, v, repeat))
                    for v in ('baseline', 'mhc')]
            assert {k['name'] for k in pair[0]['attention_kernels']} == {k['name'] for k in pair[1]['attention_kernels']}
        rows.append(dict(width=width, depth=16, heads=width//128, **variants,
            extra_step_ms=mhc['mean_ms']-base['mean_ms'],
            overhead_pct=100*(mhc['mean_ms']/base['mean_ms']-1), paired_overhead_pct=paired,
            forward_backward_event_overhead_pct=100*(mhc['forward_backward_event_ms']/base['forward_backward_event_ms']-1)))
    return dict(complete=True, job=result['job'], source=result['source'], checks=result['checks'],
        gpu=trials[0]['config']['gpu'], gpu_uuid=trials[0]['config']['gpu_uuid'],
        common_microbatch=micro, grad_accum=accum, rows=rows, trial_details=details,
        all_nvidia_trace_verified=True, matched_attention_verified=True,
        total_completed_optimizer_tokens=result['total_completed_optimizer_tokens'],
        limitations=result['protocol']['limitations'])


if __name__ == '__main__':
    root = Path(sys.argv[1]); result = json.loads((root/'comparison.json').read_text())
    audit = analyze(result)
    archive = root/'results'/f'gate-{result["job"]}.tar'
    audit['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (Path(__file__).parent/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k: v for k, v in audit.items() if k not in ('source', 'trial_details')}, indent=2))
