#!/bin/bash
set -euo pipefail
python -m pytest -q tests/test_latent_feedback.py --junitxml=results/tests.xml
python -m scripts.connection_feedback_debug --data-dir ../data --output-dir results/profile
# Actual CLI transport: exit85 checkpoint -> same protocol/run/data position.
TASK_ARGS=(--variant baseline --streams 1 --latent-feedback --attention-backend fa2
    --steps 3 --schedule-steps 305176 --microbatch 16 --seed 42 --data-dir ../data
    --run-id feedback-local-debug --segment-seconds 1800)
set +e
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../resume-check --segment-steps 2
TASK_STATUS=$?
set -e
test "$TASK_STATUS" -eq 85
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../resume-check --resume auto
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../continuous-check
python -m scripts.connection_feedback_resume_check --resumed ../resume-check \
    --continuous ../continuous-check --data-dir ../data --output results/cli-resume.json
rm -r ../resume-check ../continuous-check
echo 'FEEDBACK_DEBUG_COMPLETE'
