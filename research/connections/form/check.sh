#!/bin/bash
set -euo pipefail
python -m pytest -q tests/test_form_plan.py tests/test_form.py tests/test_latent_feedback.py --junitxml=results/form-tests.xml
python -m scripts.connection_form_debug --data-dir ../data --output-dir results/form-profile
# Real production shapes, real CCE/FA2, and an actual CLI restart after Form #1.
TASK_ARGS=(--variant baseline --streams 1 --form taylor --form-count 10
    --form-ema-decay 0.99 --form-lambda 1.0 --attention-backend fa2
    --steps 110 --schedule-steps 305176 --microbatch 16 --seed 42 --data-dir ../data
    --run-id form-local-debug --segment-seconds 1800 --output-dir ../form-resume-check)
set +e
python -m scripts.connection_train "${TASK_ARGS[@]}" --segment-steps 10
TASK_STATUS=$?
set -e
test "$TASK_STATUS" -eq 85
set +e
python -m scripts.connection_train "${TASK_ARGS[@]}" --resume auto --segment-steps 1
TASK_STATUS=$?
set -e
test "$TASK_STATUS" -eq 85
python -m scripts.connection_form_resume_check --directory ../form-resume-check --output results/form-cli-resume.json
rm -r ../form-resume-check
echo FORM_VALIDATION_COMPLETE
