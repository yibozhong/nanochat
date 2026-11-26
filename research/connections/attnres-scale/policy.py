"""GPU policy for attnres-scale."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.connections.gpu_policy import (
    HOPPER, RANK, SCALE_COMMON as COMMON, SCALE_FALLBACK as FALLBACK,
    desired_fallback, scale_gpus as require_gpus,
)
