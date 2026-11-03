#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
mkdir -p results raw cache
trap 'tar -cf "$TASK_ROOT/results.tar" -C "$TASK_ROOT" results' EXIT
export PATH="/opt/project/.venv/bin:$PATH"
export NANOCHAT_DTYPE=bfloat16 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 RAYON_NUM_THREADS=8
sha256sum -c inputs.sha256
tar -xf nanochat-145da40.tar
tar -xf tokenizer.tar -C cache
mv ./*.parquet raw/
for mode in native text_only indexed4 indexed8; do
    python benchmark_pack.py --mode "$mode" --output "results/round1-$mode.json"
done
for mode in indexed8 indexed4 text_only native; do
    python benchmark_pack.py --mode "$mode" --output "results/round2-$mode.json"
done
python benchmark_pack.py --mode native --steps 160 --warmup 32 --profile --output results/profile-native.json
python - <<'PY'
import json
from pathlib import Path
rows=[json.loads(p.read_text()) for p in sorted(Path('results').glob('round*.json'))]
assert len(rows)==8
for field in ('train_prefix_sha256','source_states_sha256','dev_sha256','test_sha256','hostname','cpu_model'):
    assert len({r[field] for r in rows})==1,field
summary={mode:sum(r['timed_seconds'] for r in rows if r['mode']==mode)/2 for mode in ('native','text_only','indexed4','indexed8')}
result=dict(complete=True,logging_backend='local',byte_and_source_state_parity=True,trials=rows,
    average_timed_seconds=summary,speedup_vs_native={k:summary['native']/v for k,v in summary.items()})
Path('results/benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='trials'}),flush=True)
PY
