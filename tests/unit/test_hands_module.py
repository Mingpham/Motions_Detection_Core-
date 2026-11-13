from pathlib import Path

import numpy as np
import pytest

from movecore.core.config import ConfigError
from movecore.core.module import FrameContext
from movecore.hands import HandsModule
from movecore.hands import module as hands_module
from movecore.hands.landmarks import Hand

FAST = {"start_frames": 2, "stop_frames": 3, "stable_frames": 2}


class FakeDetector:
    """Stands in for MediaPipe: returns the hands scripted for each frame, in order."""

    script: list[list[Hand]] = []
    created: list["FakeDetector"] = []

    def __init__(self, model_path, **options):
        self.model_path, self.options = model_path, options
        self.frames = iter(FakeDetector.script)
        self.closed = False
        FakeDetector.created.append(self)

    def detect(self, frame, timestamp_ms):
        return next(self.frames)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    for name in ("gesture_recognizer", "hand_landmarker"):
        Path("models").mkdir(exist_ok=True)
        Path(f"models/{name}.task").write_bytes(b"model")
    monkeypatch.setattr(hands_module, "HandDetector", FakeDetector)
    FakeDetector.created.clear()


@pytest.fixture
def hand(hand_points):
    """Build a Hand as the detector would return it."""

    def make(side="Right", fingers="1111", gesture=None, gesture_score=0.9, score=0.95):
        world = hand_points(fingers)
        image = np.clip(world + (0.5, 0.7, 0), 0, 1)  # somewhere inside the frame
        return Hand(side, score, image, world, gesture, gesture_score)

    return make


def run(script, options=FAST):
    FakeDetector.script = script
    module = HandsModule(options)
    module.setup()
    frame = np.zeros((480, 640, 3), np.uint8)
    results = [module.process(FrameContext(frame, i, i * 100)) for i in range(len(script))]
    events = [(e.type, e.payload) for r in results for e in r.events]
    return module, results, events


def test_hand_enters_and_leaves(hand):
    _, _, events = run([[hand()]] * 3 + [[]] * 3, FAST | {"gestures": {"rules": False}})

    assert events == [
        ("hand.enter", {"hand": "Right"}),
        ("hand.leave", {"hand": "Right", "duration_ms": 400}),
    ]


def test_gestures_from_rules_and_recognizer(hand):
    _, results, events = run([[hand(fingers="1100", gesture="victory")]] * 3)

    assert events == [
        ("hand.enter", {"hand": "Right"}),
        ("gesture.detected", {"hand": "Right", "gesture": "fingers_3", "method": "rules"}),
        (
            "gesture.detected",
            {"hand": "Right", "gesture": "victory", "method": "builtin", "score": 0.9},
        ),
    ]
    reading = results[-1].data["hands"][0]
    assert reading.fingers == (True, True, True, False, False)


def test_held_gesture_is_reported_once(hand):
    _, _, events = run([[hand()]] * 10)
    assert [e for e in events if e[0] == "gesture.detected"] == [
        ("gesture.detected", {"hand": "Right", "gesture": "fingers_5", "method": "rules"})
    ]


def test_a_new_gesture_is_reported(hand):
    _, _, events = run([[hand(fingers="1111")]] * 3 + [[hand(fingers="1000")]] * 2)
    gestures = [e[1]["gesture"] for e in events if e[0] == "gesture.detected"]
    assert gestures == ["fingers_5", "fingers_2"]  # thumb and index


def test_brief_dropout_does_not_leave_or_repeat_the_gesture(hand):
    _, _, events = run([[hand()]] * 3 + [[]] * 2 + [[hand()]] * 3)
    assert [e[0] for e in events] == ["hand.enter", "gesture.detected"]


def test_gesture_is_reported_again_after_the_hand_comes_back(hand):
    _, _, events = run([[hand()]] * 3 + [[]] * 3 + [[hand()]] * 3)
    gestures = [e[1]["gesture"] for e in events if e[0] == "gesture.detected"]
    assert gestures == ["fingers_5", "fingers_5"]


def test_unsure_recognizer_labels_are_ignored(hand):
    _, results, events = run([[hand(gesture="open_palm", gesture_score=0.3)]] * 3)

    assert results[-1].data["hands"][0].builtin_gesture is None
    assert all(e[1].get("method") != "builtin" for e in events)


def test_rules_can_be_turned_off(hand):
    _, results, events = run([[hand()]] * 3, FAST | {"gestures": {"rules": False}})

    assert results[-1].data["hands"][0].rule_gesture is None
    assert [e[0] for e in events] == ["hand.enter"]


def test_two_hands_are_tracked_separately(hand):
    _, _, events = run([[hand("Left"), hand("Right", fingers="0000")]] * 3)
    entered = sorted(e[1]["hand"] for e in events if e[0] == "hand.enter")
    assert entered == ["Left", "Right"]


def test_same_side_twice_counts_the_surer_hand(hand):
    unsure = hand("Right", fingers="0000", score=0.6)
    sure = hand("Right", fingers="1111", score=0.9)

    _, results, events = run([[sure, unsure]] * 3)

    assert len(results[-1].data["hands"]) == 2  # both are still drawn
    assert [e[1]["gesture"] for e in events if e[0] == "gesture.detected"] == ["fingers_5"]


def test_detector_gets_the_options_and_is_closed():
    module, _, _ = run([[]], FAST | {"max_hands": 1, "gestures": {"builtin_recognizer": False}})
    module.close()

    detector = FakeDetector.created[0]
    assert detector.model_path == Path("models/hand_landmarker.task")
    assert detector.options["max_hands"] == 1
    assert detector.options["recognizer"] is False
    assert detector.closed


def test_missing_model_is_reported_before_setup():
    Path("models/gesture_recognizer.task").unlink()
    with pytest.raises(ConfigError, match="Hand model not found"):
        HandsModule({})


def test_draw_marks_the_hand(hand):
    module, results, _ = run([[hand(gesture="open_palm")]])
    frame = np.zeros((480, 640, 3), np.uint8)

    module.draw(frame, results[0])

    assert frame.any()
