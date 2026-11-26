"""Submit the historical layer-feedback training recipe."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.connections.feedback_submit import submit_feedback


def main():
    submit_feedback('layer_feedback')


if __name__ == '__main__':
    main()
