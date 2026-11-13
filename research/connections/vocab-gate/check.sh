#!/bin/bash
set -euo pipefail
export WANDB_MODE=disabled
python -m pytest -q tests/test_vocab_gate.py --junitxml=results/gpu-tests.xml
python research/connections/vocab-gate/benchmark.py
bash research/connections/vocab-gate/check_cli.sh
python research/connections/vocab-gate/approve_gate.py --report-only
