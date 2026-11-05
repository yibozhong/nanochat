"""Exact review boundary for hardware selection added while the A100 gate ran."""
import subprocess
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
GATE_SOURCE='7daac76e21d5e4298fa262ddb922d979aa775566'
# No model, mixer, NVIDIA operation, optimizer or dependency changes allowed.
RUNTIME_POLICY_PATHS={
    'nanochat/connection_experiment.py',
    'scripts/connection_train.py',
    'scripts/connection_context_profile.py',
    'scripts/connection_stream_mlp_gate.py',
    'scripts/connection_stream_mlp_preflight.py',
}


def check_policy_update(gate_commit,policy_commit):
    assert gate_commit==GATE_SOURCE,gate_commit
    changed=subprocess.check_output(['git','diff','--name-only',gate_commit,policy_commit,'--',
        'nanochat','scripts','research/connections/runtime','research/connections/toolchain'],cwd=REPO,text=True).splitlines()
    assert set(changed)<=RUNTIME_POLICY_PATHS,changed
    return dict(gpu_validated_model_commit=gate_commit,policy_commit=policy_commit,
        changed_runtime_policy_paths=changed,
        invariant='All model, mixer, official NVIDIA ops, optimizer, precision and dependency files unchanged',
        additional_checks='Formal worker tests hardware-selected backend plus full-batch finite updates before NTP training')
