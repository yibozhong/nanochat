#!/bin/bash
set -euo pipefail
GRADFORMER_WORKDIR="$PWD"
mkdir -p results
trap 'tar -cf "$GRADFORMER_WORKDIR/results.tar" -C "$GRADFORMER_WORKDIR" results' EXIT
export NETRC="$PWD/.netrc" WANDB_DIR="$PWD/results"
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none
export OMP_NUM_THREADS=1 NANOCHAT_ATTENTION_BACKEND=sdpa
cp /opt/gradformer-environment.txt results/environment.txt
cp /opt/gradformer-locks.txt results/environment-locks.txt
tar -xzf llama2-balance-v1.tar.gz
tar -xzf nanochat-balance-v1.tar.gz
tar -xf dataset-v1.tar
export NANOCHAT_DTYPE=float32
cd llama2.c
python -m gradformer.check_balance --backend llama2 --device cuda --output ../results/llama2-balance-check.json
cd ../nanochat
python -m gradformer.check_balance --backend nanochat --device cuda --output ../results/nanochat-balance-check.json
cd ../llama2.c
export NANOCHAT_DTYPE=bfloat16
python -m gradformer.benchmark --backend llama2 --depth 6 --width 288 --heads 6 \
    --batch 16 --sequence 256 --accumulation 8 --include-balance --output ../results/balance-benchmark.json
python - <<'VERIFY_COST'
import json
from pathlib import Path
result=json.loads(Path('../results/balance-benchmark.json').read_text())
row=next(r for r in result['results'] if r['backward']=='balanced-reuse')
assert row['slowdown_vs_baseline'] < 10, row
assert row['balance/active'] == 1
assert abs(row['balance/aux_to_ntp_ratio']-1) < 2e-5
VERIFY_COST
common=(--backend llama2 --update adam --variant adam-balanced-ratio1
        --output ../results/adam-balanced-ratio1 --data ../dataset
        --depth 6 --width 288 --heads 6 --sequence 256 --batch 16
        --batch-tokens 32768 --tokens 200000000 --seed 1337
        --eval-every 250 --eval-batches 32 --save-every 1000
        --backward reuse --aux-weight 1 --alpha 1 --beta 1
        --gradient-balance global --balance-ratio 1)
python -m gradformer.train "${common[@]}" --stop-after-step 50
python - <<'VERIFY_PILOT'
import json
import math
from pathlib import Path
folder=Path('../results/adam-balanced-ratio1')
rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
assert json.loads((folder/'latest.json').read_text())['step']==50
assert rows[-1]['step']==50 and math.isfinite(rows[-1]['val_ce'])
measured=[r for r in rows if 'balance/aux_to_ntp_ratio' in r]
assert len(measured)>=6
for r in measured:
    assert r['balance/active']==1
    assert abs(r['balance/aux_to_ntp_ratio']-1)<2e-5
    assert all(math.isfinite(v) for v in r.values() if isinstance(v,(int,float)))
print('Verified first 50 full-batch updates: measured auxiliary/NTP block norm ratio = 1.', flush=True)
VERIFY_PILOT
python -m gradformer.train "${common[@]}" --resume
python - <<'VERIFY_FINAL'
import json
import math
from pathlib import Path
folder=Path('../results/adam-balanced-ratio1')
done=json.loads((folder/'completed.json').read_text())
assert done['tokens']==200015872 and math.isfinite(done['final_val_ce'])
rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
measured=[r for r in rows if 'balance/aux_to_ntp_ratio' in r]
assert all(r['balance/active']==1 and abs(r['balance/aux_to_ntp_ratio']-1)<2e-5 for r in measured)
print('Completed balanced 200M-token run; all logged norm ratios verified.', flush=True)
VERIFY_FINAL
