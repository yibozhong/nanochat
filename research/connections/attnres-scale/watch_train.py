"""Monitor the attnres-scale training job."""
from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.connections.train_monitor import first_marked_step, watch_scale

VARIANT = 'attnres_scale'
first_formal_step = partial(first_marked_step, variant=VARIANT)


def main(root):
    watch_scale(root, VARIANT, 'AttnResScaleTrainKey', 'attnres-scale-10b-20261108', first_formal_step)


if __name__ == '__main__':
    main(sys.argv[1])
