"""Observe startup of the attnres training job."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.connections.train_monitor import first_step_after_preflight as first_formal_step, watch_start


def main(root):
    watch_start(root, first_formal_step)


if __name__ == '__main__':
    main(sys.argv[1])
