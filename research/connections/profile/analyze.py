"""Derive the overhead audit from the returned profiling archive (stdlib only)."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with tarfile.open(args.archive) as tar:
        result = json.load(tar.extractfile('results/profile.json'))
        traces = {}
        for name in result['variants']:
            raw = tar.extractfile(f'results/{name}/trace.json.gz').read()
            events = json.loads(gzip.decompress(raw))['traceEvents']
            kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
            adds = [e for e in kernels if 'CUDAFunctor_add<c10::BFloat16>' in e['name']]
            cpu = {e['args']['External id']: e for e in events
                   if e.get('cat') == 'cpu_op' and 'External id' in e.get('args', {})}
            parents = Counter()
            for kernel in adds:
                op = cpu[kernel['args']['External id']]
                enclosing = [e for e in events if e.get('ph') == 'X'
                             and e.get('tid') == op['tid']
                             and e.get('name', '').startswith('autograd::engine::evaluate_function:')
                             and e['ts'] <= op['ts']
                             and e['ts'] + e['dur'] >= op['ts'] + op['dur']]
                assert len(enclosing) == 1, enclosing
                parents[enclosing[0]['name']] += 1
            steps = result['variants'][name]['cuda_profile']['profile_steps']
            traces[name] = dict(compressed_trace_sha256=hashlib.sha256(raw).hexdigest(),
                bf16_add_ms_per_step=sum(e['dur'] for e in adds) / 1000 / steps,
                bf16_add_count_per_step=len(adds) / steps,
                bf16_add_autograd_parents=dict(parents))
    variants = result['variants']
    baseline, mhc, no_manifold = [variants[n] for n in ('baseline', 'mhc', 'mhc_no_manifold')]
    wall_delta = mhc['normal_ms']['mean'] - baseline['normal_ms']['mean']
    kernel_delta = mhc['cuda_profile']['sum_kernel_ms_per_step'] - baseline['cuda_profile']['sum_kernel_ms_per_step']
    backward = mhc['cuda_profile']['categories']['connection_projection_backward_gemm']['ms_per_step']
    analysis = dict(source=result['source'], parent_run_id=result['run_id'],
        archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        mhc_minus_baseline_wall_ms=wall_delta,
        mhc_overhead_percent=100 * wall_delta / baseline['normal_ms']['mean'],
        mhc_minus_baseline_profiled_kernel_ms=kernel_delta,
        projection_backward_share_of_added_kernel_time_percent=100 * backward / kernel_delta,
        remove_sinkhorn_wall_ms=mhc['normal_ms']['mean'] - no_manifold['normal_ms']['mean'],
        finite_check_removal_wall_ms={n: v['normal_ms']['mean'] - v['no_finite_check_ms']['mean']
                                     for n, v in variants.items()},
        traces=traces,
        notes=['Kernel attribution uses two profiled steps; ordinary timing uses 20 uninstrumented steps.',
               'CUDA event intervals include stream gaps; CPU finite-check time includes waiting for backward.',
               'No kernel rewrite, production finite-check removal, or long pretraining was performed.'])
    args.output.write_text(json.dumps(analysis, indent=2) + '\n')


if __name__ == '__main__':
    main()
