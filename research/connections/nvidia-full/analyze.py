"""Require NVIDIA-only mHC traces and produce the full-pipeline comparison."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def main():
    root=Path(sys.argv[1]);result=json.loads((root/'comparison.json').read_text())
    assert result['complete'] and result['wandb_verified']
    trials=result['trials'];base=trials['baseline']['normal_ms']['mean']
    partial=trials['mhc_nvidia_projection_only']['normal_ms']['mean']
    original=trials['mhc_deepseek']['normal_ms']['mean']
    assert len({v['config']['gpu_uuid'] for v in trials.values()})==1
    expected={'_mhc_projection_fwd_fused':34,'_mhc_projection_bwd_fused_dx':34,
        '_mhc_scale_fwd_fused':34,'_mhc_scale_bwd_fused':34,
        '_mhc_aggregate_fwd':34,'_mhc_aggregate_bwd':34,
        '_mhc_expand_combine_fwd':32,'_mhc_expand_combine_bwd':32,
        '_mhc_sinkhorn_fwd_fused':32,'_mhc_sinkhorn_bwd_fused':32}
    rows={}
    for name,v in trials.items():
        kernels=v['cuda_profile']['kernels'];ms=v['normal_ms']['mean'];counts={}
        if name.startswith('mhc_nvidia_full'):
            assert not any(any(f in k['name'] for f in ('mhc_pre_','mhc_post','mhc_head','expand_to_mhc')) for k in kernels)
            for fragment,n in expected.items():
                counts[fragment]=sum(k['count'] for k in kernels if fragment in k['name'])
                assert counts[fragment]==n,(name,fragment,counts[fragment],n)
            assert all('_fused' in k['name'] for k in kernels if 'sinkhorn' in k['name'].lower())
        rows[name]=dict(mean_ms=ms,median_ms=v['normal_ms']['median'],stdev_ms=v['normal_ms']['stdev'],
            overhead_vs_baseline_pct=100*(ms/base-1),reduction_vs_projection_only_pct=100*(1-ms/partial),
            reduction_vs_deepseek_pct=100*(1-ms/original),
            memory_gib={k:b/1024**3 for k,b in v['normal_timing_memory'].items()},
            nvidia_kernel_counts=counts,phase_means=v['phase_means'],run_id=v['run_id'])
    archive=root/'results'/f'gate-{result["job"]}.tar'
    with tarfile.open(archive) as tar:
        errors={p.name:json.load(tar.extractfile(p)) for p in tar.getmembers()
            if p.isfile() and p.name.endswith('.json') and
            (p.name.startswith('results/oracle-controlled/') or p.name.startswith('results/oracle-native/pipeline-'))}
    audit=dict(job=result['job'],source=result['source'],checks=result['checks'],timings=rows,
        gpu=trials['baseline']['config']['gpu'],gpu_uuid=trials['baseline']['config']['gpu_uuid'],
        all_nvidia_mhc_trace_verified=True,numerical_errors=errors,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        limitations=['Short throughput diagnostics, not a quality comparison',
            'NVIDIA uses its native RMS epsilon, sigmoid and log-space Sinkhorn arithmetic',
            'Persistent interleaved stream layout requires permuting existing phi columns',
            'Embedding stream replication is standard PyTorch; all five mHC operators and final head use NVIDIA APIs',
            'Peak memory is PyTorch allocator, not total device usage'])
    dest=Path(__file__).parent; (dest/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('numerical_errors','source')},indent=2))


if __name__=='__main__':main()
