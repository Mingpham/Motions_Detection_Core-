"""MediaPipe's hand models behind one small class, so the rest of MoveCore never touches them.

Uses the Tasks API in VIDEO mode: frames go in one at a time with increasing timestamps, and
MediaPipe tracks hands between frames instead of searching the whole image every time.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from movecore.hands.landmarks import Hand

SWAP_SIDE = {"Left": "Right", "Right": "Left"}
# The Gesture Recognizer's 7 labels, renamed to match the rule-based ones. Its "None"
# (no known gesture) isn't listed, so it becomes None.
BUILTIN_GESTURES = {
    "Closed_Fist": "closed_fist",
    "Open_Palm": "open_palm",
    "Pointing_Up": "pointing_up",
    "Thumb_Down": "thumb_down",
    "Thumb_Up": "thumb_up",
    "Victory": "victory",
    "ILoveYou": "i_love_you",
}


class HandDetector:
    """Finds up to ``max_hands`` hands per frame, each with 21 landmarks.

    With ``recognizer=True`` it loads the Gesture Recognizer, which runs the same hand
    landmarker internally and adds one of 7 gesture labels per hand. Otherwise it loads the
    plain Hand Landmarker, which is a little faster.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        recognizer: bool,
        max_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        mirrored: bool,
    ) -> None:
        # Imported here, not at the top: it takes ~1 s, which motion-only runs shouldn't pay.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._recognizer = recognizer
        self._mirrored = mirrored
        self._last_timestamp = -1
        common: dict[str, Any] = {
            "base_options": BaseOptions(model_asset_path=str(model_path)),
            "running_mode": vision.RunningMode.VIDEO,
            "num_hands": max_hands,
            "min_hand_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        }
        if recognizer:
            options = vision.GestureRecognizerOptions(**common)
            self._task = vision.GestureRecognizer.create_from_options(options)
        else:
            options = vision.HandLandmarkerOptions(**common)
            self._task = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray, timestamp_ms: int) -> list[Hand]:
        """Hands in a BGR frame. Timestamps must increase; repeats are nudged forward."""
        # VIDEO mode rejects a timestamp that isn't larger than the last one.
        timestamp_ms = max(timestamp_ms, self._last_timestamp + 1)
        self._last_timestamp = timestamp_ms
        # OpenCV frames are BGR; MediaPipe expects RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self._recognizer:
            result = self._task.recognize_for_video(image, timestamp_ms)
        else:
            result = self._task.detect_for_video(image, timestamp_ms)
        return hands_from_result(result, self._mirrored)

    def close(self) -> None:
        self._task.close()


def hands_from_result(result: Any, mirrored: bool) -> list[Hand]:
    """Turn a MediaPipe result into ``Hand`` objects.

    MediaPipe (1.2.2) names the side as seen on an image that isn't mirrored, although its
    docs say the opposite. Checked both ways: on an ordinary photo it names the hands
    correctly, and on the mirrored camera view it called Ming's right hand "Left". So on a
    mirrored image the side is swapped back.
    """
    gestures = getattr(result, "gestures", None)  # only the Gesture Recognizer has these
    hands = []
    for i, landmarks in enumerate(result.hand_landmarks):
        side = result.handedness[i][0]
        handedness = SWAP_SIDE[side.category_name] if mirrored else side.category_name
        gesture, gesture_score = None, 0.0
        if gestures and gestures[i]:
            top = gestures[i][0]
            gesture = BUILTIN_GESTURES.get(top.category_name)
            gesture_score = float(top.score) if gesture else 0.0
        hands.append(
            Hand(
                handedness=handedness,
                score=float(side.score),
                image=_points(landmarks),
                world=_points(result.hand_world_landmarks[i]),
                gesture=gesture,
                gesture_score=gesture_score,
            )
        )
    return hands


def _points(landmarks: Any) -> np.ndarray:
    return np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)
