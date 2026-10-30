"""Same-GPU 0.94B / 1.43B / 2.01B vanilla versus full NVIDIA mHC sweep."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid
import xml.etree.ElementTree as ET

import wandb

GROUP = 'mhc-scale-1b-2b-20261030'
WIDTHS = (2048, 2560, 3072)


def stream(cmd, logpath, env, timeout=3600):
    with logpath.open('w') as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        def copy_output():
            for line in proc.stdout:
                log.write(line); log.flush(); print(line, end='', flush=True)
        reader = threading.Thread(target=copy_output, daemon=True); reader.start()
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(); reader.join(); raise
        reader.join()
        if code:
            raise subprocess.CalledProcessError(code, cmd)


def environment(variant):
    env = os.environ.copy()
    if variant == 'baseline':
        env['NVIDIA_TF32_OVERRIDE'] = '0'
    else:
        env.pop('NVIDIA_TF32_OVERRIDE', None)
    return env


def main():
    out = Path('results'); out.mkdir(exist_ok=True)
    source = json.loads(Path('source.json').read_text())
    protocol = json.loads(Path('research/connections/scale/protocol.json').read_text())
    ident = uuid.uuid4().hex[:8]
    kwargs = dict(project='nanochat', entity='yibozhong657-none', group=GROUP,
        name='1b-to-2b-width-sweep', job_type='correctness-and-profiling', id=ident,
        config=dict(source=source, protocol=protocol))
    try:
        run = wandb.init(**kwargs, settings=wandb.Settings(init_timeout=60))
    except wandb.errors.CommError:
        run = wandb.init(**kwargs, mode='offline')
    result = dict(complete=False, source=source, run_id=ident, protocol=protocol,
                  probes={}, trials={}, pretraining_launched=False)
    def save():
        p = out / 'comparison.tmp'; p.write_text(json.dumps(result, indent=2)+'\n')
        p.replace(out / 'comparison.json')
    def worker(width, variant, micro, repetition=0, probe=False):
        name = f'{"probe" if probe else "profile"}-d{width}-{variant}-r{repetition}-mb{micro}'
        path = out / name
        cmd = [sys.executable, '-m', 'scripts.connection_scale_profile', '--width', str(width),
               '--variant', variant, '--microbatch', str(micro), '--repetition', str(repetition),
               '--output-dir', str(path)]
        if probe:
            cmd.append('--probe')
        stream(cmd, out / (name+'.log'), environment(variant))
        v = json.loads((path / 'summary.json').read_text())
        assert v['complete'] and v['source'] == source
        result['probes' if probe else 'trials'][name] = v
        save()
        return v
    try:
        save()
        # New widths change official kernel tiling. Test their mathematics and
        # real 2K model causality before any capacity or throughput measurement.
        stream([sys.executable, '-m', 'pytest', 'tests/test_mhc_nvidia_scale.py', '-v',
                '--junitxml=results/scale-tests.xml'], out/'scale-tests.log', environment('mhc'))
        suites = ET.parse(out/'scale-tests.xml').getroot().findall('testsuite')
        checks = {k: sum(int(s.attrib.get(k, 0)) for s in suites)
                  for k in ('tests', 'failures', 'errors', 'skipped')}
        assert checks == dict(tests=9, failures=0, errors=0, skipped=0), checks
        result['checks'] = checks; save()
        manifest = json.loads(Path('../data/manifest.json').read_text())
        for name, info in manifest['files'].items():
            digest = hashlib.sha256()
            with Path('../data', name+'.bin').open('rb') as file:
                while chunk := file.read(8*1024*1024):
                    digest.update(chunk)
            assert digest.hexdigest() == info['sha256'], name
        result['data_manifest'] = manifest; save()
        micro = None
        for candidate in (16, 8, 4, 2, 1):
            mhc = worker(3072, 'mhc', candidate, probe=True)
            if mhc['status'] != 'fits':
                continue
            baseline = worker(3072, 'baseline', candidate, probe=True)
            if baseline['status'] == 'fits':
                micro = candidate; break
        if micro is None:
            raise RuntimeError('No common microbatch fits with memory headroom; no timings were run')
        result['common_microbatch'] = micro; save()
        run.summary['common_microbatch'] = micro
        print(f'Common microbatch={micro}, accumulation={16//micro}', flush=True)
        # Reverse both size order and paired variant order on the second pass.
        # Every trial is freshly initialized in a separate process. Repeats
        # measure timing variability, not independent quality-training seeds.
        for repetition in (0, 1):
            widths = WIDTHS if repetition == 0 else WIDTHS[::-1]
            for width in widths:
                variants = ('baseline', 'mhc') if repetition == 0 else ('mhc', 'baseline')
                for variant in variants:
                    v = worker(width, variant, micro, repetition)
                    assert v['status'] == 'complete' and v['optimizer_updates'] == 86
                    run.log({f'd{width}/{variant}/step_ms': v['normal_ms']['mean'], 'repetition': repetition})
        assert len(result['trials']) == 12
        assert len({v['config']['gpu_uuid'] for v in result['trials'].values()}) == 1
        result['total_completed_optimizer_tokens'] = sum(
            v['training_tokens'] for group in ('probes', 'trials') for v in result[group].values())
        result['complete'] = True; save()
        run.summary.update(dict(complete=True, checks_passed=True, trials=12,
            total_completed_optimizer_tokens=result['total_completed_optimizer_tokens']))
        run.save('results/*.json', base_path='.', policy='now'); run.finish()
    except BaseException:
        save(); run.finish(exit_code=1); raise


if __name__ == '__main__':
    main()
