"""Audit the completed GPU gate and publish before formal training can start."""
import fcntl
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile

REPO=Path(__file__).resolve().parents[4]
ROOT=Path(sys.argv[1]).resolve()


def command(args):
    return subprocess.check_output(args,cwd=REPO,text=True,timeout=60).strip()


def main():
    with (ROOT/'publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'published.json').exists():return
        state=json.loads((ROOT/'state.json').read_text())
        assert state['status']=='complete'
        result=json.loads((ROOT/'results.json').read_text())
        assert result['complete'] and result['checks']==dict(tests=17,failures=0,errors=0,skipped=0)
        assert result['source']==state['source'] and result['logging_backend']=='local' and not result['formal_training']
        assert len(result['trials'])==6 and len({r['gpu_uuid'] for r in result['trials']})==1
        for r in result['trials']:
            assert r['complete'] and r['fits_22gb_budget'] and len(r['step_seconds'])==30
            assert r['causal_prefix_max_abs']==r['batch_isolation_max_abs']==0
            assert r['batch_tokens']==32768 and r['microbatch']==16 and r['grad_accum']==1
        assert result['summary']['shared']['parameters']==result['summary']['per_stream']['parameters']
        diagnostics={}
        with tarfile.open(ROOT/'results'/f'stream-{state["job"]}.tar') as tf:
            assert not any(m.name.endswith(('.pt','.wandb')) for m in tf)
            for name in ('stream-eager-compiled','shared-eager-compiled-control','stream-compiled-fp32-loop'):
                diagnostics[name]=json.load(tf.extractfile(f'results/nvidia-full-checks/{name}.json'))
        for name in ('stream-eager-compiled','shared-eager-compiled-control'):
            d=diagnostics[name]
            assert d['passed'] and d['global_relative_l2']<.02
            assert all(p['finite'] and p['mixed_l2_ratio']<=1 for p in d['parameters'].values())
        assert max(diagnostics['stream-compiled-fp32-loop']['gradients'])<3e-6
        assert not command(['git','status','--porcelain'])
        # Current changes may only add orchestration/reporting; GPU-tested model,
        # training loop and dependency versions must still be the same.
        assert not command(['git','diff','--name-only',state['source']['commit'],'HEAD','--',
            'nanochat','scripts','research/connections/runtime','research/connections/toolchain'])
        report=REPO/'research/connections/stream-attention/validation.json'
        audited=dict(result,job=state['job'],numerical_diagnostics=diagnostics)
        content=json.dumps(audited,indent=2)+'\n'
        if not report.exists() or report.read_text()!=content:
            report.write_text(content)
            command(['git','add',str(report)])
            command(['git','commit','-m','Record verified multistream attention correctness and GPU throughput'])
        head=command(['git','rev-parse','HEAD'])
        try:
            runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']()
        except Exception:
            # The PTY helper can race a successful child exit; remote is decisive.
            assert command(['git','ls-remote','origin','refs/heads/main']).split()[0]==head
        assert command(['git','ls-remote','origin','refs/heads/main']).split()[0]==head
        marker=dict(gate_commit=state['source']['commit'],remote_commit=head,job=state['job'])
        temp=ROOT/'published.tmp';temp.write_text(json.dumps(marker,indent=2)+'\n');temp.replace(ROOT/'published.json')
        print(json.dumps(marker),flush=True)


if __name__=='__main__':main()
