"""Options for the motion module, read from ``modules.motion`` in the config."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from movecore.core.options import OptionChecks
from movecore.motion.zones import Zone, parse_zones

METHODS = ("diff", "mog2", "knn")
CHECKS = OptionChecks("modules.motion")


@dataclass(frozen=True)
class MotionSettings:
    method: str = "mog2"
    threshold: int = 25
    blur: int = 21
    history: int = 500
    var_threshold: float = 16.0
    detect_shadows: bool = True
    background_ratio: float = 0.9
    kernel_size: int = 5
    min_area: int = 500
    warmup_frames: int = 30
    start_frames: int = 2
    stop_frames: int = 15
    zones: tuple[Zone, ...] = ()

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "MotionSettings":
        CHECKS.reject_unknown(options, {field.name for field in fields(cls)} | {"enabled"})
        values = {key: value for key, value in options.items() if key != "enabled"}
        values["zones"] = parse_zones(values.get("zones"))
        settings = cls(**values)
        settings._validate()
        return settings

    def _validate(self) -> None:
        CHECKS.choice("method", self.method, METHODS)
        CHECKS.integer("threshold", self.threshold, 0, 255)
        CHECKS.odd("blur", self.blur)
        CHECKS.integer("history", self.history, 1)
        CHECKS.odd("kernel_size", self.kernel_size)
        CHECKS.integer("min_area", self.min_area, 0)
        CHECKS.integer("warmup_frames", self.warmup_frames, 0)
        CHECKS.integer("start_frames", self.start_frames, 1)
        CHECKS.integer("stop_frames", self.stop_frames, 1)
        CHECKS.number("var_threshold", self.var_threshold)
        CHECKS.number("background_ratio", self.background_ratio, at_most=1)
        CHECKS.boolean("detect_shadows", self.detect_shadows)
