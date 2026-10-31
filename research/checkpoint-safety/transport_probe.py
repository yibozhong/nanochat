"""Small CPU-only payload to check real HTCondor checkpoint/eviction transport."""
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

state_path = Path('checkpoints/probe.pkl')
if state_path.exists():
    state = pickle.loads(state_path.read_bytes())
else:
    state = dict(step=0, data_row=0, value=0., run_id='transport-probe', rng=random.Random(42).getstate())
restored = state['step']
assert state['data_row'] == state['step']*16 and state['run_id'] == 'transport-probe'
rng = random.Random()
rng.setstate(state['rng'])
ad = Path(os.environ['_CONDOR_JOB_AD']).read_text()
eviction_issued = any(line.strip().lower() == 'transportevictionissued = true' for line in ad.splitlines())
print(json.dumps(dict(event='restored', step=restored, eviction_issued=eviction_issued)), flush=True)
for step in range(restored+1, 7):
    state.update(step=step, data_row=step*16, value=state['value']+rng.random(), rng=rng.getstate())
    tmp = state_path.with_suffix('.tmp')
    tmp.write_bytes(pickle.dumps(state))
    tmp.replace(state_path)
    print(json.dumps(dict(event='step', step=step)), flush=True)
    if step == 3:
        # This exit must cause an actual scheduler upload before the eviction.
        sys.exit(85)
    if step == 4 and not eviction_issued:
        print('WAITING_FOR_CONTROLLED_EVICTION local_step=4 durable_step=3', flush=True)
        time.sleep(300)
        raise RuntimeError('Controller did not perform the required eviction')
    time.sleep(.2)
reference = random.Random(42)
expected = sum(reference.random() for _ in range(6))
assert state['value'] == expected and state['rng'] == reference.getstate()
assert eviction_issued and restored == 3
result = dict(complete=True, restored_after_eviction_step=restored, final_step=state['step'],
    lost_unuploaded_steps=1, rng_and_value_match_uninterrupted=True, stable_run_id=True,
    scope='CPU synthetic payload tests scheduler transport; model/optimizer restore is covered by prior GPU gates')
Path('results/transport-probe.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
