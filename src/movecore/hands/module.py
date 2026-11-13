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

    def close(self) -> None:
        self._detector.close()

    def process(self, ctx: FrameContext) -> ModuleResult:
        readings = [self._read(hand) for hand in self._detector.detect(ctx.frame, ctx.timestamp_ms)]
        # Events are kept per side. If both hands are given the same side, the surer one counts.
        by_side = {reading.hand.handedness: reading for reading in sorted(readings, key=_score)}
        events = []
        for side in SIDES:
            events += self._update_side(side, by_side.get(side), ctx)
        return ModuleResult(data={"hands": readings}, events=events)

    def draw(self, frame: np.ndarray, result: ModuleResult) -> None:
        for reading in result.data["hands"]:
            draw_hand(frame, reading.hand, reading.fingers, _describe(reading))

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
