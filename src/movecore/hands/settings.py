"""Options for the hands module (``modules.hands`` in the config), checked once at startup."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from movecore.core.config import ConfigError
from movecore.core.options import OptionChecks

CHECKS = OptionChecks("modules.hands")
GESTURE_CHECKS = OptionChecks("modules.hands.gestures")
# Gesture methods that later milestones add; until then they must stay off.
NOT_BUILT_YET = {"custom_classifier": "M5", "dynamic": "M6"}
DOWNLOAD_HINT = "download the models with: uv run python scripts/download_models.py"


@dataclass(frozen=True)
class GestureSettings:
    rules: bool = True
    builtin_recognizer: bool = True
    custom_classifier: bool = False
    dynamic: bool = False

    @classmethod
    def from_options(cls, options: Any) -> "GestureSettings":
        if not isinstance(options, Mapping):
            raise CHECKS.error("gestures", "a mapping", options)
        GESTURE_CHECKS.reject_unknown(options, {f.name for f in fields(cls)})
        settings = cls(**options)
        for f in fields(cls):
            GESTURE_CHECKS.boolean(f.name, getattr(settings, f.name))
        for name, milestone in NOT_BUILT_YET.items():
            if getattr(settings, name):
                raise ConfigError(
                    f"'modules.hands.gestures.{name}' arrives in {milestone}; set it to false"
                )
        return settings


@dataclass(frozen=True)
class HandSettings:
    max_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    mirrored: bool = True
    pinch_ratio: float = 0.25
    min_gesture_score: float = 0.5
    start_frames: int = 3
    stop_frames: int = 10
    stable_frames: int = 5
    landmarker_model: str = "models/hand_landmarker.task"
    recognizer_model: str = "models/gesture_recognizer.task"
    gestures: GestureSettings = field(default_factory=GestureSettings)

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "HandSettings":
        CHECKS.reject_unknown(options, {f.name for f in fields(cls)} | {"enabled"})
        values = {key: value for key, value in options.items() if key != "enabled"}
        values["gestures"] = GestureSettings.from_options(values.get("gestures", {}))
        settings = cls(**values)
        # Events are kept per hand, and a hand is known by its side, so two at most.
        CHECKS.integer("max_hands", settings.max_hands, 1, 2)
        CHECKS.number("min_detection_confidence", settings.min_detection_confidence, at_most=1)
        CHECKS.number("min_tracking_confidence", settings.min_tracking_confidence, at_most=1)
        CHECKS.boolean("mirrored", settings.mirrored)
        CHECKS.number("pinch_ratio", settings.pinch_ratio, at_most=1)
        CHECKS.number("min_gesture_score", settings.min_gesture_score, at_most=1)
        CHECKS.integer("start_frames", settings.start_frames, 1)
        CHECKS.integer("stop_frames", settings.stop_frames, 1)
        CHECKS.integer("stable_frames", settings.stable_frames, 1)
        for name in ("landmarker_model", "recognizer_model"):
            value = getattr(settings, name)
            if not (isinstance(value, str) and value.endswith(".task")):
                raise CHECKS.error(name, "a path to a .task model file", value)
        return settings

    @property
    def model_path(self) -> Path:
        """The one model to load: the recognizer already contains a full hand landmarker."""
        if self.gestures.builtin_recognizer:
            return Path(self.recognizer_model)
        return Path(self.landmarker_model)

    def check_model(self) -> Path:
        path = self.model_path
        if not path.is_file():
            raise ConfigError(f"Hand model not found: {path}; {DOWNLOAD_HINT}")
        return path
