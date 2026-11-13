import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from movecore.hands.dataset import DatasetError, load_samples
from movecore.hands.training import evaluate, format_confusion, to_classifier, train


def test_distinct_poses_are_learned(write_gestures):
    dataset = load_samples(write_gestures())

    classifier, evaluations, best = train(dataset)

    assert [e.name for e in evaluations] == ["logistic_regression", "neural_network"]
    assert best.accuracy > 0.95
    assert classifier.labels == ("fist", "open", "peace")
    label, score = classifier.predict(dataset.features[0])
    assert label == dataset.labels[0]


def test_whole_sessions_are_held_out_when_there_are_two(write_gestures):
    two = evaluate(load_samples(write_gestures(sessions=2)), "lr", LogisticRegression)
    one = evaluate(load_samples(write_gestures(sessions=1, folder="one")), "lr", LogisticRegression)

    assert two.by_session
    assert not one.by_session
    assert two.confusion.sum() == 180  # every sample is tested exactly once


@pytest.mark.parametrize(
    "model",
    [
        LogisticRegression(max_iter=2000),
        MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=0),
    ],
)
@pytest.mark.parametrize("labels", [("open", "fist", "peace"), ("open", "fist")])
def test_numpy_copy_matches_scikit_learn(write_gestures, model, labels):
    # Two gestures is a special case: scikit-learn stores one score instead of two.
    dataset = load_samples(write_gestures(labels=labels, samples=20))
    model.fit(dataset.features, dataset.labels)

    classifier = to_classifier(model)

    expected = model.predict_proba(dataset.features)
    assert classifier.probabilities(dataset.features) == pytest.approx(expected, abs=1e-4)


def test_unsupported_models_are_refused():
    with pytest.raises(TypeError):
        to_classifier(object())
    with pytest.raises(ValueError, match="ReLU"):
        to_classifier(MLPClassifier(activation="tanh"))


def test_one_gesture_is_not_enough(write_gestures):
    with pytest.raises(DatasetError, match="at least 2 gestures"):
        train(load_samples(write_gestures(labels=("open",))))


def test_too_few_samples_are_named(write_gestures):
    with pytest.raises(DatasetError, match="too few"):
        train(load_samples(write_gestures(sessions=1, samples=5)))


def test_confusion_table_lists_every_gesture(write_gestures):
    best = evaluate(load_samples(write_gestures()), "lr", LogisticRegression)
    table = format_confusion(best).splitlines()

    assert len(table) == 4
    assert table[0].split()[-3:] == ["fist", "open", "peace"]
    assert table[1].split()[0] == "fist"
    assert np.trace(best.confusion) == pytest.approx(best.confusion.sum() * best.accuracy)
