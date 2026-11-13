"""Train and evaluate gesture classifiers with scikit-learn.

Only the training script imports this; the live app runs the result with NumPy alone
(``classifier.py``).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier

from movecore.hands.classifier import GestureClassifier
from movecore.hands.dataset import Dataset, DatasetError

MIN_SAMPLES_PER_GESTURE = 20
MAX_FOLDS = 5

# Two candidates, simplest first: on a tie, the simpler one wins.
CANDIDATES: dict[str, Callable[[], Any]] = {
    # One layer: a weighted sum of the 63 features per gesture, then softmax.
    "logistic_regression": lambda: LogisticRegression(max_iter=2000),
    # One hidden layer of 64 ReLU units first, so it can learn combinations of features.
    "neural_network": lambda: MLPClassifier(
        hidden_layer_sizes=(64,), max_iter=2000, random_state=0
    ),
}


@dataclass(frozen=True)
class Evaluation:
    name: str
    accuracy: float
    labels: tuple[str, ...]
    confusion: np.ndarray  # row = true gesture, column = predicted gesture
    by_session: bool  # True when tested on recording runs the model never saw


def check_dataset(dataset: Dataset) -> None:
    counts = dataset.counts()
    if len(counts) < 2:
        raise DatasetError(f"Need at least 2 gestures to tell apart, found {list(counts)}")
    small = {label: n for label, n in counts.items() if n < MIN_SAMPLES_PER_GESTURE}
    if small:
        raise DatasetError(
            f"Need at least {MIN_SAMPLES_PER_GESTURE} samples per gesture, too few: {small}"
        )


def evaluate(dataset: Dataset, name: str, make_model: Callable[[], Any]) -> Evaluation:
    """Cross-validation: train on part of the data, test on the rest, rotate, repeat.

    Samples from one recording run are near-copies of each other. If copies land on both
    the training and the test side, the test is too easy (data leakage). So when every
    gesture has at least 2 runs, whole runs are held out together.
    """
    sessions = min(dataset.session_counts().values())
    if sessions >= 2:
        folds: Any = StratifiedGroupKFold(n_splits=min(MAX_FOLDS, sessions))
        groups = dataset.sessions
    else:
        smallest = min(dataset.counts().values())
        folds = StratifiedKFold(n_splits=min(MAX_FOLDS, smallest), shuffle=True, random_state=0)
        groups = None
    predicted = cross_val_predict(
        make_model(), dataset.features, dataset.labels, cv=folds, groups=groups
    )
    labels = tuple(sorted(dataset.counts()))
    return Evaluation(
        name=name,
        accuracy=float(np.mean(predicted == dataset.labels)),
        labels=labels,
        confusion=confusion_matrix(dataset.labels, predicted, labels=list(labels)),
        by_session=groups is not None,
    )


def train(dataset: Dataset) -> tuple[GestureClassifier, list[Evaluation], Evaluation]:
    """Evaluate every candidate, fit the best one on all the data, and return it with the
    evaluations and the best one's evaluation."""
    check_dataset(dataset)
    evaluations = [evaluate(dataset, name, make) for name, make in CANDIDATES.items()]
    best = max(evaluations, key=lambda evaluation: evaluation.accuracy)  # first wins a tie
    model = CANDIDATES[best.name]()
    model.fit(dataset.features, dataset.labels)
    return to_classifier(model), evaluations, best


def to_classifier(model: Any) -> GestureClassifier:
    """Copy a fitted LogisticRegression or MLPClassifier's numbers into a GestureClassifier."""
    if isinstance(model, LogisticRegression):
        weights, biases = [model.coef_.T], [model.intercept_]
    elif isinstance(model, MLPClassifier):
        if model.activation != "relu":
            raise ValueError(f"Only ReLU hidden layers are supported, got {model.activation}")
        weights, biases = list(model.coefs_), list(model.intercepts_)
    else:
        raise TypeError(f"Unsupported model: {type(model).__name__}")
    if weights[-1].shape[1] == 1:
        # With 2 gestures, scikit-learn keeps one score z and uses sigmoid(z) for the second.
        # softmax([0, z]) gives exactly the same probabilities.
        weights[-1] = np.hstack([np.zeros_like(weights[-1]), weights[-1]])
        biases[-1] = np.concatenate([np.zeros(1), biases[-1]])
    return GestureClassifier(
        labels=tuple(str(label) for label in model.classes_),
        weights=tuple(np.asarray(w, dtype=np.float32) for w in weights),
        biases=tuple(np.asarray(b, dtype=np.float32) for b in biases),
    )


def format_confusion(evaluation: Evaluation) -> str:
    """The confusion matrix as a text table."""
    labels = evaluation.labels
    header = "true \\ predicted"
    first = max(len(header), *(len(label) for label in labels)) + 2
    width = max(len(label) for label in labels) + 2
    lines = [header.ljust(first) + "".join(label.rjust(width) for label in labels)]
    for label, row in zip(labels, evaluation.confusion, strict=True):
        lines.append(label.ljust(first) + "".join(str(n).rjust(width) for n in row))
    return "\n".join(lines)
