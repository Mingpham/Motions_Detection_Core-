"""The hands module: 21 landmarks per hand, gestures, and ``hand.*`` / ``gesture.*`` events."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from movecore.core.config import ConfigError
from movecore.core.debounce import Debouncer, StableLabel
from movecore.core.events import Event
from movecore.core.module import FrameContext, Module, ModuleResult
from movecore.hands.classifier import ClassifierError, GestureClassifier
from movecore.hands.dynamic import PathTracker, Swipe, SwipeDetector, hand_size_px, palm_center
from movecore.hands.features import hand_features
from movecore.hands.gesture_rules import extended_fingers, rule_gesture
from movecore.hands.landmarker import HandDetector
from movecore.hands.landmarks import CONNECTIONS, TIPS, WRIST, Hand
from movecore.hands.settings import TRAIN_HINT, HandSettings
from movecore.output.overlay import draw_label

SIDES = ("Left", "Right")
METHODS = ("rules", "builtin", "custom")
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
YELLOW = (0, 220, 255)
GRAY = (160, 160, 160)
# One color per fingertip trail (BGR): thumb red, index magenta, middle cyan, ring orange,
# pinky green.
FINGER_COLORS = {
    "thumb": (0, 0, 255),
    "index": (255, 0, 255),
    "middle": (255, 255, 0),
    "ring": (0, 165, 255),
    "pinky": (0, 255, 0),
}
# How long a swipe's label stays on screen after the swipe.
SWIPE_LABEL_MS = 600


@dataclass(frozen=True)
class HandReading:
    """One hand plus what MoveCore made of it."""

    hand: Hand
    fingers: tuple[bool, ...]  # extended or not: thumb, index, middle, ring, pinky
    rule_gesture: str | None  # None when the rules are off
    builtin_gesture: str | None  # None when the recognizer is off or not sure enough
    custom_gesture: str | None = None  # None when the classifier is off or not sure enough
    custom_score: float = 0.0


class HandsModule(Module):
    """Tracks up to two hands and reports when they appear, leave, or change gesture.

    ``result.data["hands"]`` is a list of ``HandReading``. Events: ``hand.enter`` and
    ``hand.leave`` (with ``duration_ms``) per side, and ``gesture.detected`` with ``hand``,
    ``gesture`` and ``method`` ('rules', 'builtin' or 'custom'; the last two add ``score``).
    With ``gestures.dynamic``, also ``gesture.swipe`` with ``hand``, ``direction``,
    ``distance`` and ``speed`` (hand sizes, per second) and ``duration_ms``; ``result.data``
    and ``fingers`` (every finger that swiped that way); ``result.data`` then holds
    ``paths`` (palm path per side, for later uses), ``trails`` (side -> finger -> fingertip
    path, what swipes are judged on) and ``swipes`` (side -> recent "direction (fingers)").
    """

    name = "hands"

    def __init__(self, options: Mapping[str, Any]) -> None:
        self.settings = HandSettings.from_options(options)
        # Checked now, so a missing model is reported before the camera opens.
        self._model = self.settings.check_model()
        self._classifier: GestureClassifier | None = None
        if self.settings.gestures.custom_classifier:
            try:
                self._classifier = GestureClassifier.load(self.settings.classifier_model)
            except ClassifierError as exc:
                raise ConfigError(f"{exc}; {TRAIN_HINT}") from exc

    def setup(self) -> None:
        s = self.settings
        self._detector = HandDetector(
            self._model,
            recognizer=s.gestures.builtin_recognizer,
            max_hands=s.max_hands,
            min_detection_confidence=s.min_detection_confidence,
            min_tracking_confidence=s.min_tracking_confidence,
            mirrored=s.mirrored,
        )
        self._present = {side: Debouncer(s.start_frames, s.stop_frames) for side in SIDES}
        self._since = dict.fromkeys(SIDES, 0)
        self._labels = {side: self._new_labels() for side in SIDES}
        self._swipes = {
            side: SwipeDetector(
                s.swipe_window_ms, s.swipe_min_distance, s.swipe_cooldown_ms, s.swipe_fingers
            )
            for side in SIDES
        }
        self._last_swipe: dict[str, tuple[str, int]] = {}  # side -> (direction, timestamp)
        self._paths = {side: PathTracker(s.path_window_ms) for side in SIDES}

    def close(self) -> None:
        self._detector.close()

    def process(self, ctx: FrameContext) -> ModuleResult:
        readings = [self._read(hand) for hand in self._detector.detect(ctx.frame, ctx.timestamp_ms)]
        # Events are kept per side. If both hands are given the same side, the surer one counts.
        by_side = {reading.hand.handedness: reading for reading in sorted(readings, key=_score)}
        events = []
        for side in SIDES:
            events += self._update_side(side, by_side.get(side), ctx)
        data: dict[str, Any] = {"hands": readings}
        if self.settings.gestures.dynamic:
            data["paths"] = {side: path.trail for side, path in self._paths.items()}
            data["trails"] = {side: detector.trails for side, detector in self._swipes.items()}
            data["swipes"] = {
                side: direction
                for side, (direction, at) in self._last_swipe.items()
                if ctx.timestamp_ms - at <= SWIPE_LABEL_MS
            }
        return ModuleResult(data=data, events=events)

    def draw(self, frame: np.ndarray, result: ModuleResult) -> None:
        for path in result.data.get("paths", {}).values():
            _polyline(frame, path, GRAY, 1)
        for fingers in result.data.get("trails", {}).values():
            for finger, trail in fingers.items():
                _polyline(frame, trail, FINGER_COLORS[finger], 3)
        swipes = result.data.get("swipes", {})
        for reading in result.data["hands"]:
            text = _describe(reading)
            if reading.hand.handedness in swipes:
                text += f" | swipe {swipes[reading.hand.handedness]}"
            draw_hand(frame, reading.hand, reading.fingers, text)

    def _read(self, hand: Hand) -> HandReading:
        s = self.settings
        rule = rule_gesture(hand.world, s.pinch_ratio) if s.gestures.rules else None
        sure = hand.gesture is not None and hand.gesture_score >= s.min_gesture_score
        custom, custom_score = None, 0.0
        if self._classifier is not None:
            label, score = self._classifier.predict(hand_features(hand.world, hand.looks_like))
            if score >= s.min_classifier_score:
                custom, custom_score = label, score
        return HandReading(
            hand,
            extended_fingers(hand.world),
            rule,
            hand.gesture if sure else None,
            custom,
            custom_score,
        )

    def _update_side(
        self, side: str, reading: HandReading | None, ctx: FrameContext
    ) -> list[Event]:
        events = []
        change = self._present[side].update(reading is not None)
        if change == "start":
            self._since[side] = ctx.timestamp_ms
            events.append(self._event("hand.enter", ctx, {"hand": side}))
        elif change == "stop":
            duration = ctx.timestamp_ms - self._since[side]
            events.append(self._event("hand.leave", ctx, {"hand": side, "duration_ms": duration}))
            # Forget the last gesture, so showing it again after coming back is reported.
            self._labels[side] = self._new_labels()
            self._swipes[side].reset()
            self._paths[side].clear()

        # A hand missing for a frame or two keeps its gesture rather than flickering to None.
        if reading is None or not self._present[side].active:
            return events
        labels = {
            "rules": (reading.rule_gesture, None),
            "builtin": (reading.builtin_gesture, reading.hand.gesture_score),
            "custom": (reading.custom_gesture, reading.custom_score),
        }
        for method, (label, score) in labels.items():
            stable = self._labels[side][method]
            if stable.update(label) and stable.current is not None:
                payload: dict[str, Any] = {"hand": side, "gesture": label, "method": method}
                if score is not None:
                    payload["score"] = round(score, 2)
                events.append(self._event("gesture.detected", ctx, payload))
        if self.settings.gestures.dynamic:
            height, width = ctx.frame.shape[:2]
            hand = reading.hand
            size = hand_size_px(hand, width, height)
            self._paths[side].add(ctx.timestamp_ms, palm_center(hand, width, height), size)
            swipe = self._swipes[side].update(
                hand, reading.fingers, ctx.timestamp_ms, width, height
            )
            if swipe is not None:
                label = f"{swipe.direction} ({'+'.join(swipe.fingers)})"
                self._last_swipe[side] = (label, ctx.timestamp_ms)
                events.append(self._event("gesture.swipe", ctx, _swipe_payload(side, swipe)))
        return events

    def _new_labels(self) -> dict[str, StableLabel]:
        return {method: StableLabel(self.settings.stable_frames) for method in METHODS}

    def _event(self, event_type: str, ctx: FrameContext, payload: dict[str, Any]) -> Event:
        return Event(event_type, self.name, ctx.index, ctx.timestamp_ms, payload)


def draw_hand(frame: np.ndarray, hand: Hand, fingers: tuple[bool, ...], text: str) -> None:
    """Draw a hand's skeleton, ring its extended fingertips, and put ``text`` below the wrist."""
    height, width = frame.shape[:2]
    points = [tuple(p) for p in hand.pixels(width, height).tolist()]
    for a, b in CONNECTIONS:
        cv2.line(frame, points[a], points[b], WHITE, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 3, GREEN, cv2.FILLED)
    for extended, tip in zip(fingers, TIPS, strict=True):
        if extended:
            cv2.circle(frame, points[tip], 7, YELLOW, 2)
    # Below the wrist, kept inside the frame.
    x, y = points[WRIST]
    origin = (min(max(x - 40, 6), width - 160), min(max(y + 30, 20), height - 10))
    draw_label(frame, text, origin)


def _polyline(frame: np.ndarray, trail: list[tuple[float, float]], color, thickness: int) -> None:
    points = np.array(trail, dtype=np.int32)
    if len(points) > 1:
        cv2.polylines(frame, [points], False, color, thickness, cv2.LINE_AA)


def _swipe_payload(side: str, swipe: Swipe) -> dict[str, Any]:
    return {
        "hand": side,
        "direction": swipe.direction,
        "fingers": list(swipe.fingers),
        "distance": round(swipe.distance, 1),
        "speed": round(swipe.speed, 1),
        "duration_ms": swipe.duration_ms,
    }


def _score(reading: HandReading) -> float:
    return reading.hand.score


def _describe(reading: HandReading) -> str:
    parts = [reading.hand.handedness]
    if reading.rule_gesture:
        parts.append(reading.rule_gesture)
    if reading.builtin_gesture:
        parts.append(f"{reading.builtin_gesture} {reading.hand.gesture_score:.2f}")
    if reading.custom_gesture:
        parts.append(f"{reading.custom_gesture} {reading.custom_score:.2f}")
    return " | ".join(parts)
