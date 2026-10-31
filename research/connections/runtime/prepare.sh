#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
mkdir -p results runtime
trap 'tar -cf "$TASK_ROOT/preparation-results.tar" -C "$TASK_ROOT" results' EXIT
export PATH="/opt/project/.venv/bin:$PATH"
export UV_CACHE_DIR="$PWD/uv-cache" UV_NO_CACHE=1
cp pyproject.toml uv.lock runtime/
uv sync --frozen --no-build-package transformer-engine-torch --project runtime --python /opt/project/.venv/bin/python
uv pip freeze --python runtime/.venv/bin/python > results/environment.txt
find runtime/.venv -name nvcc -o -name cuda_runtime.h > results/cuda-paths.txt
runtime/.venv/bin/python - <<'PYINFO'
import json,importlib.metadata as m
from pathlib import Path
info={p:m.version(p) for p in ['torch','transformer-engine','cut-cross-entropy']}
Path('results/runtime.json').write_text(json.dumps(info,indent=2))
print(info)
PYINFO
tar -czf runtime.tar.gz runtime
sha256sum runtime.tar.gz > results/runtime.sha256
