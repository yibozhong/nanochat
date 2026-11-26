"""GPU policy for latent-feedback."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.connections.gpu_policy import (
    HOPPER, RANK, FEEDBACK_COMMON as COMMON, FEEDBACK_FALLBACK as FALLBACK,
    desired_fallback, feedback_gpus as require_gpus,
)
