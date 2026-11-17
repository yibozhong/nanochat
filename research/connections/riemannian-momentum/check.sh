#!/bin/bash
set -euo pipefail
TASK_RESIDUAL_TEXT=$(python research/connections/riemannian-momentum/experiments.py "$RMT_EXPERIMENT")
mapfile -t TASK_RESIDUAL_ARGS <<< "$TASK_RESIDUAL_TEXT"
python -m pytest -q tests/test_riemannian_momentum.py tests/test_latent_feedback.py tests/test_residual_variants.py --junitxml=results/rmt-tests.xml
python -m scripts.connection_riemannian_debug --residual-dynamics "$TASK_RESIDUAL_MODE" --velocity-init "$TASK_VELOCITY_INIT" --data-dir ../data --output-dir results/rmt-profile
TASK_ARGS=(--variant baseline --streams 1 "${TASK_RESIDUAL_ARGS[@]}" --attention-backend fa2
    --steps 3 --schedule-steps 305176 --microbatch 16 --seed 42 --data-dir ../data
    --run-id rmt-local-debug --segment-seconds 1800)
set +e
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../rmt-resume-check --segment-steps 2
TASK_STATUS=$?
set -e
test "$TASK_STATUS" -eq 85
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../rmt-resume-check --resume auto
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../rmt-continuous-check
python -m scripts.connection_feedback_resume_check --resumed ../rmt-resume-check \
    --continuous ../rmt-continuous-check --data-dir ../data --output results/rmt-cli-resume.json
rm -r ../rmt-resume-check ../rmt-continuous-check
echo 'RMT_DEBUG_COMPLETE'
