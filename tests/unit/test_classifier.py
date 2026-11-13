import numpy as np
import pytest

from movecore.hands.classifier import ClassifierError, GestureClassifier


def tiny(labels=("a", "b", "c")):
    """A one-layer classifier: gesture i scores feature i."""
    weights = np.zeros((63, len(labels)), np.float32)
    for i in range(len(labels)):
        weights[i, i] = 10
    return GestureClassifier(tuple(labels), (weights,), (np.zeros(len(labels), np.float32),))


def test_probabilities_add_up_to_one_and_pick_the_highest_score():
    features = np.zeros(63)
    features[1] = 1

    probabilities = tiny().probabilities(features)[0]
    label, score = tiny().predict(features)

    assert probabilities.sum() == pytest.approx(1)
    assert label == "b"
    assert score == pytest.approx(probabilities[1])
    assert score > 0.99


def test_hidden_layers_use_relu():
    # Hidden unit = ReLU(feature 0): negative inputs are cut to 0, so both scores tie.
    hidden = np.zeros((63, 1), np.float32)
    hidden[0, 0] = 1
    out = np.array([[1, -1]], np.float32)
    model = GestureClassifier(("a", "b"), (hidden, out), (np.zeros(1), np.zeros(2)))
    negative = np.zeros(63)
    negative[0] = -5

    assert model.probabilities(negative)[0] == pytest.approx([0.5, 0.5])


def test_large_scores_do_not_overflow():
    probabilities = tiny().probabilities(np.full(63, 1000.0))
    assert np.isfinite(probabilities).all()


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "models" / "clf.npz"
    tiny().save(path)

    loaded = GestureClassifier.load(path)

    assert path.is_file()  # saved under the exact name, not "clf.npz.npz"
    assert loaded.labels == ("a", "b", "c")
    features = np.random.default_rng(0).normal(size=63)
    assert loaded.probabilities(features) == pytest.approx(tiny().probabilities(features))


@pytest.mark.parametrize(
    ("weights", "biases", "labels", "message"),
    [
        ((np.zeros((62, 2)),), (np.zeros(2),), ("a", "b"), "Layer 0"),
        ((np.zeros((63, 3)),), (np.zeros(3),), ("a", "b"), "3 outputs for 2 labels"),
        ((np.zeros((63, 2)),), (np.zeros(3),), ("a", "b"), "Layer 0"),
        ((np.zeros((63, 1)),), (np.zeros(1),), ("a",), "at least 2"),
        ((), (), ("a", "b"), "weights and a bias"),
    ],
)
def test_inconsistent_shapes_are_rejected(weights, biases, labels, message):
    with pytest.raises(ClassifierError, match=message):
        GestureClassifier(labels, weights, biases)


def test_missing_file(tmp_path):
    with pytest.raises(ClassifierError, match="not found"):
        GestureClassifier.load(tmp_path / "nope.npz")


def test_pickled_content_is_refused(tmp_path):
    # An object array can only be stored with pickle, which could run code when loaded.
    path = tmp_path / "evil.npz"
    with path.open("wb") as file:
        np.savez(file, version=1, labels=np.array([{"a": 1}], dtype=object))

    with pytest.raises(ClassifierError, match="not a valid gesture classifier"):
        GestureClassifier.load(path)


def test_wrong_version_and_garbage_are_refused(tmp_path):
    old = tmp_path / "old.npz"
    with old.open("wb") as file:
        np.savez(file, version=99, labels=np.array(["a", "b"]))
    garbage = tmp_path / "garbage.npz"
    garbage.write_bytes(b"not a zip file")

    with pytest.raises(ClassifierError, match="version 99"):
        GestureClassifier.load(old)
    with pytest.raises(ClassifierError, match="not a valid"):
        GestureClassifier.load(garbage)
