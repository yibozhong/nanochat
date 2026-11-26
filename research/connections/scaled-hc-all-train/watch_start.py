"""Observe startup of the scaled-hc-all-train training job."""
from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.connections.train_monitor import first_marked_step, watch_start

first_formal_step = partial(first_marked_step, variant='scaled_hc_all')


def main(root):
    watch_start(root, first_formal_step)


if __name__ == '__main__':
    main(sys.argv[1])
