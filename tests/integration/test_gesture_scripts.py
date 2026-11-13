"""Record → train → use: the two M5 scripts, the way a person runs them."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from movecore.hands import HandsModule
from movecore.hands.landmarks import Hand

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDetector:
    """Always sees one right hand, in the pose given by ``FakeDetector.world``."""

    world = None

    def __init__(self, model_path, **options):
        pass

    def detect(self, frame, timestamp_ms):
        return [Hand("Right", 0.9, np.full((21, 3), 0.5), FakeDetector.world)]

    def close(self):
        pass


class FakeDisplay:
    """Presses SPACE on the first frame, then lets the recording run to the end."""

    def __init__(self, title):
        self.frames = 0
        self.last_key = -1

    def show(self, frame):
        self.frames += 1
        self.last_key = ord(" ") if self.frames == 1 else -1
        return self.frames < 200

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        pass


@pytest.fixture
def recorder(monkeypatch, make_video, tmp_path):
    """The record script wired to a video file, a fake window and a fake hand detector."""
    script = load_script("record_gestures")
    monkeypatch.setattr(script, "HandDetector", FakeDetector)
    monkeypatch.setattr(script, "Display", FakeDisplay)
    Path("models").mkdir()
    Path("models/hand_landmarker.task").write_bytes(b"model")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"input": {"width": 160, "height": 120}, "modules": {}, "output": {}})
    )
    clip = make_video(frames=120)
    return lambda *args: script.main([*args, "--config", str(config), "--source", str(clip)])


def test_record_train_and_recognize(recorder, hand_points, capsys):
    poses = {"open": hand_points("1111"), "fist": hand_points("0000", thumb_out=False)}
    for label, world in poses.items():
        FakeDetector.world = world
        assert recorder(label, "--samples", "25") == 0
    assert "Saved 25 samples of 'fist'" in capsys.readouterr().out

    trainer = load_script("train_gesture_classifier")
    assert trainer.main([]) == 0
    out = capsys.readouterr().out
    assert "optimistic" in out  # one session per gesture
    assert "Saved" in out and Path("models/gesture_clf.npz").is_file()

    hands = HandsModule({"gestures": {"builtin_recognizer": False, "custom_classifier": True}})
    reading = hands._read(Hand("Right", 0.9, np.zeros((21, 3)), poses["fist"]))
    assert reading.custom_gesture == "fist"


@pytest.mark.parametrize("label", ["../escape", "Rock"])
def test_record_rejects_unsafe_labels(recorder, label, capsys):
    assert recorder(label) == 2
    assert "Gesture label" in capsys.readouterr().err
    assert not Path("data").exists()


def test_train_without_data_explains(capsys):
    assert load_script("train_gesture_classifier").main([]) == 2
    assert "record_gestures.py" in capsys.readouterr().err


def test_train_refuses_a_pickle_output(capsys):
    assert load_script("train_gesture_classifier").main(["--output", "clf.joblib"]) == 2
