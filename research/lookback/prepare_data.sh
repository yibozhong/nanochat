#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
mkdir -p results raw cache
trap 'tar -cf "$TASK_ROOT/results.tar" -C "$TASK_ROOT" results' EXIT
export PATH="/opt/project/.venv/bin:$PATH"
export NANOCHAT_DTYPE=bfloat16 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 RAYON_NUM_THREADS=4
sha256sum -c prepare-inputs.sha256
tar -xf nanochat-145da40.tar
tar -xf tokenizer.tar -C cache
mv ./*.parquet raw/
cp /opt/gradformer-environment.txt results/container-environment.txt
cp prepare-inputs.sha256 results/
python prepare_data.py
tar -cf data.tar data cache/tokenizer cache/tokenizer-provenance.json
tar -cf gate-data.tar -C gate-data data
python - <<'PY'
import hashlib,json
from pathlib import Path
for name in ('data.tar','gate-data.tar'):
    p=Path(name);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    Path('results/'+p.stem+'-archive.json').write_text(json.dumps(dict(bytes=p.stat().st_size,sha256=h.hexdigest()),indent=2)+'\n')
PY
