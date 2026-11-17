#!/bin/bash
set -euo pipefail
python -m pytest -q tests/test_transport_momentum.py tests/test_nonzero_residual.py --junitxml=results/momentum-tests.xml
python -m scripts.connection_momentum_debug --data-dir ../data --output-dir results/momentum-profile --nonzero-init --transport-rms
TASK_ARGS=(--variant baseline --streams 1 --residual-dynamics ngpt_heavy_ball
    --ngpt-velocity-mode transport_rms --momentum-velocity-rms-floor .1
    --residual-projection-init normal --ngpt-force-rms-floor 1e-6
    --velocity-init token --attention-backend fa2 --steps 3 --schedule-steps 305176
    --microbatch 16 --seed 42 --data-dir ../data --run-id momentum-local-debug --segment-seconds 1800)
set +e
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../momentum-resume-check --segment-steps 2
TASK_STATUS=$?
set -e
test "$TASK_STATUS" -eq 85
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../momentum-resume-check --resume auto
python -m scripts.connection_train "${TASK_ARGS[@]}" --output-dir ../momentum-continuous-check
python -m scripts.connection_feedback_resume_check --resumed ../momentum-resume-check \
    --continuous ../momentum-continuous-check --data-dir ../data --output results/momentum-cli-resume.json
rm -r ../momentum-resume-check ../momentum-continuous-check
echo 'MOMENTUM_DEBUG_COMPLETE'
