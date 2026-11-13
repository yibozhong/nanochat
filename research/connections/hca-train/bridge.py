"""Transfer a validated interactive first segment to its existing held run."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from policy import STEPS, validate_shared
from submit import write


def read_member(archive, name):
    member = archive.getmember(name)
    assert member.isfile() and member.size < 32 * 1024**2
    return archive.extractfile(member).read()


def validate_archives(results, checkpoints, state):
    """Inspect metadata only; never deserialize tensor checkpoints on the AP."""
    with tarfile.open(results) as tf:
        assert int(read_member(tf, 'results/debug-requests/' + state['bridge']['request'] + '.exit')) == 85
        progress = json.loads(read_member(tf, 'results/train/progress.json'))
        config = json.loads(read_member(tf, 'results/train/config.json'))
        ready = json.loads(read_member(tf, 'results/hca-startup/passed.json'))
        assert json.loads(read_member(tf, 'results/source.json')) == state['source']
    assert progress['run_id'] == state['run_id'] and progress['source'] == state['source']
    assert 0 < progress['step'] < STEPS and progress['tokens'] == progress['step'] * 32768
    assert math.isfinite(progress['best_dev_ce']) and progress['training_seconds'] > 0
    assert config['source'] == state['source'] and config['logging_backend'] == 'wandb'
    assert config['model']['attention_mode'] == 'fixed_hca' and config['model']['hca_backend'] == 'triton'
    assert config['protocol']['microbatch'] == state['microbatch'] and config['protocol']['seed'] == 42
    assert config['protocol']['steps'] == config['protocol']['schedule_steps'] == STEPS
    assert ready['run_id'] == state['run_id'] and ready['attention_mode'] == 'fixed_hca'
    assert validate_shared(ready, state['source']['commit']) == state['microbatch']
    with tarfile.open(checkpoints) as tf:
        sizes = {}
        for name in ('latest.pt', 'best.pt'):
            member = tf.getmember('checkpoints/' + name)
            assert member.isfile() and member.size > 0
            sizes[name] = member.size
    return dict(progress=progress, checkpoint_bytes=sizes)


def refresh_log(root, state):
    target = root / f'logs/train.{state["job"]}.out'
    temp = target.with_suffix('.incoming')
    request = state['bridge']['request']
    subprocess.run(['condor_ssh_to_job', '-ssh', 'scp', state['job'], '-O',
                    'remote:results/debug-requests/' + request + '.log', str(temp)],
                   check=True, timeout=30, capture_output=True)
    temp.replace(target)


def handoff(root, state, ad):
    """Release exactly the existing run, only after successful durable transfer."""
    assert ad['JobStatus'] == 4 and ad.get('ExitCode') == 0 and not ad.get('ExitBySignal', False)
    bridge = state['bridge']; debug = Path(bridge['debug_root'])
    results = debug / f'results/debug-{state["job"]}.tar'
    checkpoints = debug / f'results/debug-{state["job"]}-checkpoints.tar'
    evidence = validate_archives(results, checkpoints, state)
    target = bridge['resume_job']
    ads = json.loads(subprocess.check_output(['condor_q', target, '-json'], text=True))
    assert len(ads) == 1
    held = ads[0]
    assert held['JobStatus'] == 5 and held['HoldReason'].startswith('User requested interactive debug;')
    assert held['HCARunId'] == state['run_id'] and held['HCATrainMode'] == 'fixed_hca'
    assert held['RequestGPUs'] == 1 and held['RequestCpus'] == 2
    assert held['RequestMemory'] == 32768 and held['RequestDisk'] == 80 * 1024**2
    assert held['SuccessCheckpointExitCode'] == 85
    assert set(held['TransferCheckpoint'].split(',')) == {'results.tar', 'checkpoints.tar'}
    assert held.get('NumCheckpoints', 0) == 0, 'Do not overwrite an existing scheduler checkpoint'
    assert 'H100|H200' in held['RequireGPUs'] and '78000' in held['RequireGPUs']
    assert '.netrc' in held['EncryptInputFiles'] and 'nanochat-uv-v4.sif' in held['TransferInput']
    assert json.loads((root / 'bundles/source.json').read_text()) == state['source']
    for name, src in [('bridge-results.tar', results), ('bridge-checkpoints.tar', checkpoints)]:
        link = root / 'bundles' / name
        if link.is_symlink():
            assert link.resolve() == src.resolve()
        else:
            assert not link.exists()
            link.symlink_to(src)
    # Initial archives use distinct names: later scheduler checkpoints always
    # take precedence, so a restart cannot replay this first segment again.
    wrapper = root / 'bridge-resume.sh'
    wrapper.write_text('''#!/bin/bash
set -euo pipefail
if [ ! -f checkpoints.tar ] && [ ! -f results.tar ]; then
    cp bridge-checkpoints.tar checkpoints.tar
    cp bridge-results.tar results.tar
fi
test -s checkpoints.tar && test -s results.tar
exec bash train.sh "$@"
''')
    wrapper.chmod(0o755)
    additions = ['bundles/bridge-results.tar', 'bundles/bridge-checkpoints.tar', 'train.sh']
    inputs = held['TransferInput'].split(',')
    for name in additions:
        if name not in inputs:
            inputs.append(name)
    for key, value in [('Cmd', str(wrapper)), ('TransferInput', ','.join(inputs))]:
        subprocess.run(['condor_qedit', target, key, json.dumps(value)], check=True, capture_output=True)
    updated = json.loads(subprocess.check_output(['condor_q', target, '-json'], text=True))[0]
    assert updated['JobStatus'] == 5 and updated['Cmd'] == str(wrapper)
    assert set(updated['TransferInput'].split(',')) == set(inputs)
    write(root / 'bridge-transfer-verified.json', dict(evidence=evidence, scheduler=updated,
                                                     debug_job=state['job'], time=time.time()))
    old = root / f'logs/train.{target}.out'
    if old.exists():
        backup = old.with_suffix('.pre-bridge.out')
        if not backup.exists():
            shutil.copy2(old, backup)
        old.write_text('')
    state.update(job=target, status='queued_for_resume', formal_started=True,
                 durable_step=evidence['progress']['step'], stdout_offset=0,
                 inside_formal_log=False, bridge_completed=dict(bridge, evidence=evidence))
    del state['bridge']
    write(root / 'state.json', state)
    subprocess.run(['condor_release', target], check=True, capture_output=True)
