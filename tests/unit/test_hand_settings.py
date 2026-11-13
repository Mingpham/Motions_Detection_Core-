from pathlib import Path

import pytest

from movecore.core.config import ConfigError
from movecore.hands.settings import HandSettings


def test_defaults_use_the_recognizer_model():
    settings = HandSettings.from_options({"enabled": True})

    assert settings.max_hands == 2
    assert settings.gestures.rules and settings.gestures.builtin_recognizer
    assert settings.model_path == Path("models/gesture_recognizer.task")


def test_plain_landmarker_when_the_recognizer_is_off():
    settings = HandSettings.from_options({"gestures": {"builtin_recognizer": False}})

    assert settings.model_path == Path("models/hand_landmarker.task")
    assert settings.gestures.rules  # the other gesture options keep their defaults


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_hand": 2}, "Unknown 'modules.hands' options: max_hand"),
        ({"max_hands": 3}, "max_hands"),
        ({"max_hands": True}, "max_hands"),
        ({"min_detection_confidence": 1.5}, "min_detection_confidence"),
        ({"mirrored": "yes"}, "mirrored"),
        ({"pinch_ratio": 0}, "pinch_ratio"),
        ({"stable_frames": 0}, "stable_frames"),
        ({"recognizer_model": "/etc/passwd"}, ".task model file"),
        ({"gestures": ["rules"]}, "'modules.hands.gestures' must be a mapping"),
        ({"gestures": {"rule": True}}, "Unknown 'modules.hands.gestures' options: rule"),
        ({"gestures": {"rules": 1}}, "modules.hands.gestures.rules"),
        ({"classifier_model": "models/gesture_clf.joblib"}, ".npz model file"),
        ({"min_classifier_score": 0}, "min_classifier_score"),
        ({"gestures": {"dynamic": True}}, "arrives in M6"),
    ],
)
def test_bad_options_are_rejected(options, message):
    with pytest.raises(ConfigError, match=message):
        HandSettings.from_options(options)


def test_missing_model_says_how_to_get_it():
    with pytest.raises(ConfigError, match="download_models.py"):
        HandSettings.from_options({}).check_model()


def test_existing_model_is_accepted():
    Path("models").mkdir()
    Path("models/gesture_recognizer.task").write_bytes(b"model")

    assert HandSettings.from_options({}).check_model() == Path("models/gesture_recognizer.task")
