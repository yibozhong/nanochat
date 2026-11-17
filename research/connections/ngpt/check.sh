#!/bin/bash
set -euo pipefail
python -m pytest -q tests/test_ngpt.py tests/test_momentum.py tests/test_latent_feedback.py --junitxml=results/ngpt-tests.xml
python -m scripts.connection_momentum_debug --data-dir ../data --output-dir results/ngpt-profile --target-mode ngpt
TASK_ARGS=(--variant baseline --streams 1 --residual-dynamics ngpt
    --attention-backend fa2 --steps 3 --schedule-steps 305176
    --microbatch 16 --seed 42 --data-dir ../data --run-id ngpt-local-debug --segment-seconds 1800)
set +e
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../ngpt-resume-check --segment-steps 2
TASK_STATUS=$?
set -e
test "$TASK_STATUS" -eq 85
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../ngpt-resume-check --resume auto
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../ngpt-continuous-check
python -m scripts.connection_feedback_resume_check --resumed ../ngpt-resume-check \
    --continuous ../ngpt-continuous-check --data-dir ../data --output results/ngpt-cli-resume.json
rm -r ../ngpt-resume-check ../ngpt-continuous-check
echo 'NGPT_DEBUG_COMPLETE'
