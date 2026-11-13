from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from movecore.hands.landmarker import HandDetector, hands_from_result

MODELS = Path(__file__).resolve().parents[2] / "models"


def point(x, y, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def category(name, score):
    return SimpleNamespace(category_name=name, score=score)


def fake_result(side="Right", gesture=None):
    """The shape of a MediaPipe result with one hand; ``gesture`` adds recognizer output."""
    result = SimpleNamespace(
        handedness=[[category(side, 0.97)]],
        hand_landmarks=[[point(i / 20, 0.5) for i in range(21)]],
        hand_world_landmarks=[[point(0, i / 100) for i in range(21)]],
    )
    if gesture is not None:
        result.gestures = [[category(gesture, 0.8)]]
    return result


def test_landmarks_become_arrays():
    (hand,) = hands_from_result(fake_result(), mirrored=False)

    assert hand.handedness == "Right"
    assert hand.score == pytest.approx(0.97)
    assert hand.image.shape == hand.world.shape == (21, 3)
    assert hand.image[20, 0] == pytest.approx(1.0)
    assert hand.gesture is None


def test_side_is_swapped_when_the_image_is_mirrored():
    # On the mirrored camera view, MediaPipe calls your right hand "Left".
    (hand,) = hands_from_result(fake_result("Left"), mirrored=True)
    assert hand.handedness == "Right"


@pytest.mark.parametrize(
    ("name", "expected"),
    [("Thumb_Up", "thumb_up"), ("ILoveYou", "i_love_you"), ("None", None)],
)
def test_recognizer_labels_are_renamed(name, expected):
    (hand,) = hands_from_result(fake_result(gesture=name), mirrored=True)

    assert hand.gesture == expected
    assert hand.gesture_score == (pytest.approx(0.8) if expected else 0.0)


def test_pixels_scale_to_the_frame():
    (hand,) = hands_from_result(fake_result(), mirrored=True)
    pixels = hand.pixels(640, 480)
    assert pixels[20].tolist() == [640, 240]


# The real models aren't in git or CI (they're downloaded), so these run only where they exist.
@pytest.mark.parametrize(
    ("filename", "recognizer"),
    [("gesture_recognizer.task", True), ("hand_landmarker.task", False)],
)
def test_real_model_finds_no_hand_in_an_empty_frame(filename, recognizer):
    model = MODELS / filename
    if not model.is_file():
        pytest.skip(f"{filename} not downloaded")
    detector = HandDetector(
        model,
        recognizer=recognizer,
        max_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        mirrored=True,
    )
    try:
        frame = np.zeros((480, 640, 3), np.uint8)
        assert detector.detect(frame, 0) == []
        assert detector.detect(frame, 0) == []  # a repeated timestamp is nudged, not an error
    finally:
        detector.close()
