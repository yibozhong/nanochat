"""Independent-process precision/timing follow-up after the first TE audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET

import wandb

GROUP = 'mhc-nvidia-native-20261030'
TRIALS = {
    'baseline': ('baseline', 'official', '0'),
    'mhc_official': ('mhc', 'official', '0'),
    'mhc_torch_compile': ('mhc', 'torch_compile', '0'),
    'mhc_nvidia_native': ('mhc', 'nvidia', None),
    'mhc_nvidia_fp32_dw': ('mhc', 'nvidia', '0'),
    'mhc_nvidia_precise': ('mhc', 'nvidia_precise', '0'),
}


def environment(override):
    env = os.environ.copy()
    if override is None: env.pop('NVIDIA_TF32_OVERRIDE', None)
    else: env['NVIDIA_TF32_OVERRIDE'] = override
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--micro-backend')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.micro_backend:
        from scripts.connection_nvidia_compare import projection_benchmark
        value = projection_benchmark((args.micro_backend,))
        value['nvidia_tf32_override'] = os.environ.get('NVIDIA_TF32_OVERRIDE')
        value['note'] = 'Fresh process for this backend; independent Triton autotuning; CUDA events include launch gaps and scalar probe. Precision is specified by backend and nvidia_tf32_override.'
        Path(args.output).write_text(json.dumps(value,indent=2)+'\n')
        return
    out = Path('results'); source = json.loads(Path('source.json').read_text())
    ident = uuid.uuid4().hex[:8]
    kwargs = dict(project='nanochat',entity='yibozhong657-none',group=GROUP,
        name='nvidia-native-precision-and-timing',job_type='correctness-and-profiling',
        id=ident,config=dict(source=source,trials=TRIALS,timed_steps=60,
            collect_compile_garbage_after_warmup=True,micro_fresh_process=True))
    try: run = wandb.init(**kwargs,settings=wandb.Settings(init_timeout=60))
    except wandb.errors.CommError: run = wandb.init(**kwargs,mode='offline')
    try:
        suites = []
        gates = [
            ('controlled', ['tests/test_mhc.py','tests/test_mhc_torch_projection.py','tests/test_mhc_nvidia_projection.py'], '0'),
            ('native', ['tests/test_mhc_nvidia_projection.py','-k','not nvidia_precise'], None),
        ]
        for label,tests,override in gates:
            print('Checks '+label,flush=True)
            with (out/f'pytest-{label}.log').open('w') as log:
                subprocess.run([sys.executable,'-m','pytest',*tests,'-q',f'--junitxml=results/tests-{label}.xml'],
                    env=environment(override),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
            suites += ET.parse(out/f'tests-{label}.xml').getroot().findall('testsuite')
            # Keep oracle values for both settings; tests use shared filenames.
            import shutil
            shutil.copytree(out/'torch-projection-checks',out/f'oracle-{label}')
        checks = {k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        assert checks == dict(tests=29,failures=0,errors=0,skipped=0),checks
        print('Checks passed '+json.dumps(checks),flush=True)
        micro = {}
        for name,(_,backend,override) in TRIALS.items():
            if name == 'baseline': continue
            print('Independent microbenchmark '+name,flush=True)
            with (out/f'micro-{name}.log').open('w') as log:
                subprocess.run([sys.executable,'-m','scripts.connection_nvidia_native',
                    '--micro-backend',backend,'--output',str(out/f'micro-{name}.json')],
                    env=environment(override),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
            micro[name] = json.loads((out/f'micro-{name}.json').read_text())
        manifest = json.loads(Path('../data/manifest.json').read_text())
        for name,info in manifest['files'].items():
            digest = hashlib.sha256()
            with Path('../data',name+'.bin').open('rb') as f:
                while chunk := f.read(8*1024*1024): digest.update(chunk)
            assert digest.hexdigest() == info['sha256']
        trials = {}
        for name,(variant,backend,override) in TRIALS.items():
            print('Profiling '+name,flush=True)
            with (out/f'{name}.log').open('w') as log:
                subprocess.run([sys.executable,'-m','scripts.connection_profile','--variant',variant,
                    '--projection-backend',backend,'--wandb-group',GROUP,'--output-dir',str(out/name),
                    '--timed-steps','60','--collect-after-warmup'],env=environment(override),
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
            v = json.loads((out/name/'summary.json').read_text())
            assert v['complete'] and v['source']==source and v['optimizer_updates']==106
            assert v['config']['nvidia_tf32_override'] == override
            trials[name] = v
            run.log({name+'/normal_step_ms':v['normal_ms']['mean']})
        result = dict(complete=True,source=source,run_id=ident,checks=checks,
            microbenchmark=micro,trials=trials,data_manifest=manifest,pretraining_launched=False)
        (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
        run.summary.update(dict(complete=True,checks_passed=True,trials=len(TRIALS),total_diagnostic_tokens=6*106*32768))
        run.save('results/*.json',base_path='.',policy='now');run.finish()
        print('Comparison complete',flush=True)
    except BaseException:
        run.finish(exit_code=1);raise


if __name__ == '__main__': main()
