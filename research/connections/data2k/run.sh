#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
mkdir -p results
trap 'tar -cf "$TASK_ROOT/results.tar" -C "$TASK_ROOT" results' EXIT
export PATH="/opt/project/.venv/bin:$PATH"
export NANOCHAT_PACKING_THREADS="${1:-4}"
bash prepare_data.sh
python finish_data.py
