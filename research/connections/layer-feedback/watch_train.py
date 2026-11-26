"""Monitor the layer-feedback training job."""
from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.connections.train_monitor import collect_feedback as collect, last_marked_step, watch_feedback

formal_step = partial(last_marked_step, variant='layer_feedback')


def main(root):
    watch_feedback(root, 'LayerFeedbackTrainKey', formal_step)


if __name__ == '__main__':
    main(sys.argv[1])
