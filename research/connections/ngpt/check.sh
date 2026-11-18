#!/bin/bash
set -euo pipefail
TASK_MODE="$1"
TASK_ARGS_EXTRA=()
TASK_WEIGHT_ARGS=()
if [[ "${2:-}" == --no-ngpt-weight-norm ]]; then TASK_WEIGHT_ARGS+=("$2"); fi
TASK_PROFILE_ARGS=()
TASK_VELOCITY_MODE=ambient
case "$TASK_MODE" in
    ngpt_hbformer)
        TASK_TESTS=tests/test_ngpt_hbformer.py
        if [[ "${2:-}" == --no-ngpt-weight-norm ]]; then
            TASK_TESTS=tests/test_ngpt_hbformer_no_wn.py
        fi
        TASK_VELOCITY_MODE=${3:-transport_rms}
        if [[ "$TASK_VELOCITY_MODE" == transport_layernorm ]]; then
            TASK_TESTS=tests/test_ngpt_velocity_layernorm.py
            TASK_PROFILE_ARGS=(--ngpt-velocity-mode "$TASK_VELOCITY_MODE")
        fi
        TASK_ARGS_EXTRA=(--velocity-init token_position --momentum-eta-init 1 --momentum-velocity-rms-floor .1)
        ;;
    ngpt_lookahead)
        TASK_TESTS=tests/test_ngpt_lookahead.py
        TASK_VELOCITY_MODE=lookahead_transport
        TASK_ARGS_EXTRA=(--velocity-init token_position --momentum-beta-init .5 --lookahead-mu-init 0)
        ;;
    heavy_ball_rms)
        TASK_TESTS=tests/test_heavy_ball_rms.py
        TASK_ARGS_EXTRA=(--velocity-init token --momentum-eta-init .05 --momentum-velocity-rms-floor .1)
        ;;
    vanilla|ngpt|ngpt_heavy_ball) TASK_TESTS=tests/test_nonzero_residual.py ;;
    *) exit 64 ;;
esac
python -m pytest -q "$TASK_TESTS" --junitxml=results/ngpt-tests.xml
python -m scripts.connection_momentum_debug --data-dir ../data --output-dir results/ngpt-profile --target-mode "$TASK_MODE" --nonzero-init "${TASK_WEIGHT_ARGS[@]}" "${TASK_PROFILE_ARGS[@]}"
TASK_ARGS=(--variant baseline --streams 1 --residual-dynamics "$TASK_MODE"
    --ngpt-velocity-mode "$TASK_VELOCITY_MODE" "${TASK_ARGS_EXTRA[@]}" "${TASK_WEIGHT_ARGS[@]}"
    --residual-projection-init normal --residual-projection-init-std .02 --ngpt-force-rms-floor 1e-6
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
