"""Train the custom gesture classifier from data/gestures/*.csv.

    uv run python scripts/train_gesture_classifier.py

Compares the candidate models with cross-validation, prints each one's accuracy and the
best one's confusion matrix, then saves the best one to models/gesture_clf.npz. Turn it on
with `custom_classifier: true` under modules.hands.gestures in the config.
"""

import argparse
import sys

from movecore.hands.dataset import DatasetError, load_samples
from movecore.hands.training import format_confusion, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the custom gesture classifier.")
    parser.add_argument("--data", default="data/gestures", help="folder of recorded CSV files")
    parser.add_argument("--output", default="models/gesture_clf.npz", help="where to save it")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.output.endswith(".npz"):
        print("error: --output must end with .npz", file=sys.stderr)
        return 2
    try:
        dataset = load_samples(args.data)
        print(f"Samples: {len(dataset)}")
        for label, count in dataset.counts().items():
            print(f"  {label:<20} {count:>5} samples, {dataset.session_counts()[label]} session(s)")
        classifier, evaluations, best = train(dataset)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not evaluations[0].by_session:
        print(
            "\nNote: a gesture has only one session, so test samples are near-copies of\n"
            "training ones and the accuracy below is optimistic. Record a second session of\n"
            "every gesture for an honest number."
        )
    print("\nCross-validated accuracy:")
    for evaluation in evaluations:
        print(f"  {evaluation.name:<20} {evaluation.accuracy:6.1%}")
    print(f"\nConfusion matrix of {best.name} (rows: true gesture, columns: predicted):")
    print(format_confusion(best))

    classifier.save(args.output)
    print(f"\nSaved {best.name} with {len(classifier.labels)} gestures to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
