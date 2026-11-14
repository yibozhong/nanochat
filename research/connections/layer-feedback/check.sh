#!/bin/bash
set -euo pipefail
python -m pytest -q tests/test_latent_feedback.py --junitxml=results/top-regression.xml
python -m pytest -q tests/test_layer_feedback.py --junitxml=results/layer-tests.xml
python -m scripts.connection_layer_feedback_debug --data-dir ../data --output-dir results/layer-profile
TASK_ARGS=(--variant baseline --streams 1 --latent-feedback --feedback-source layers
    --feedback-attention-backend triton --attention-backend fa2
    --steps 3 --schedule-steps 305176 --microbatch 16 --seed 42 --data-dir ../data
    --run-id layer-feedback-local-debug --segment-seconds 1800)
set +e
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../layer-resume-check --segment-steps 2
TASK_STATUS=$?
set -e
test "$TASK_STATUS" -eq 85
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../layer-resume-check --resume auto
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../layer-continuous-check
python -m scripts.connection_feedback_resume_check --resumed ../layer-resume-check \
    --continuous ../layer-continuous-check --data-dir ../data --output results/layer-cli-resume.json
rm -r ../layer-resume-check ../layer-continuous-check
echo 'LAYER_FEEDBACK_DEBUG_COMPLETE'
