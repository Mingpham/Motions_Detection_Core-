"""Run a trained gesture classifier with NumPy alone, from a file that can't run code.

The model is a stack of layers. Each layer multiplies by a weight matrix and adds a bias;
hidden layers then keep only positive values (ReLU). The last layer's scores go through
softmax, which turns them into probabilities that add up to 1. Logistic regression is the
same thing with no hidden layer.

Saved as ``.npz``: plain number arrays, loaded with ``allow_pickle=False``. A pickle or
joblib file can run any code when it's loaded; this format can't (ADR 0006).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from movecore.hands.features import FEATURE_COUNT

FORMAT_VERSION = 1


class ClassifierError(ValueError):
    """The classifier file is missing or invalid."""


@dataclass(frozen=True)
class GestureClassifier:
    labels: tuple[str, ...]
    weights: tuple[np.ndarray, ...]  # layer i: (inputs, outputs)
    biases: tuple[np.ndarray, ...]  # layer i: (outputs,)

    def __post_init__(self) -> None:
        if len(self.labels) < 2:
            raise ClassifierError("A classifier needs at least 2 gestures")
        if not self.weights or len(self.weights) != len(self.biases):
            raise ClassifierError("Every layer needs weights and a bias")
        inputs = FEATURE_COUNT
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            if weight.ndim != 2 or weight.shape[0] != inputs or bias.shape != (weight.shape[1],):
                raise ClassifierError(
                    f"Layer {i} has the wrong shape: {weight.shape}, {bias.shape}"
                )
            inputs = weight.shape[1]
        if inputs != len(self.labels):
            raise ClassifierError(
                f"The last layer has {inputs} outputs for {len(self.labels)} labels"
            )

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """Probability of each label, for one feature vector (63,) or many (n, 63)."""
        x = np.atleast_2d(features)
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            x = np.maximum(x @ weight + bias, 0)  # ReLU
        scores = x @ self.weights[-1] + self.biases[-1]
        # Softmax. Subtracting the largest score first keeps exp() from overflowing and
        # doesn't change the result.
        scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """The most likely label for one hand, and its probability."""
        probabilities = self.probabilities(features)[0]
        best = int(np.argmax(probabilities))
        return self.labels[best], float(probabilities[best])

    def save(self, path: str | Path) -> None:
        arrays = {f"weight_{i}": weight for i, weight in enumerate(self.weights)}
        arrays |= {f"bias_{i}": bias for i, bias in enumerate(self.biases)}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as file:  # a file object, so numpy doesn't add ".npz"
            np.savez(file, version=FORMAT_VERSION, labels=np.array(self.labels), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "GestureClassifier":
        path = Path(path)
        if not path.is_file():
            raise ClassifierError(f"Gesture classifier not found: {path}")
        try:
            with np.load(path, allow_pickle=False) as data:
                if int(data["version"]) != FORMAT_VERSION:
                    raise ClassifierError(f"{path}: unsupported format version {data['version']}")
                layers = sum(1 for name in data.files if name.startswith("weight_"))
                return cls(
                    labels=tuple(str(label) for label in data["labels"]),
                    weights=tuple(data[f"weight_{i}"].astype(np.float32) for i in range(layers)),
                    biases=tuple(data[f"bias_{i}"].astype(np.float32) for i in range(layers)),
                )
        except ClassifierError:
            raise
        except (OSError, ValueError, KeyError) as exc:
            # ValueError includes numpy refusing a file that would need pickle.
            raise ClassifierError(f"{path} is not a valid gesture classifier: {exc}") from exc
