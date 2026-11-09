"""Options for the motion module, read from ``modules.motion`` in the config."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from movecore.core.config import ConfigError
from movecore.motion.zones import Zone, parse_zones

METHODS = ("diff", "mog2", "knn")


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
        # Rejecting unknown names catches typos like `min_aera` that would silently do nothing.
        known = {field.name for field in fields(cls)} | {"enabled"}
        unknown = sorted(set(options) - known)
        if unknown:
            raise ConfigError(f"Unknown 'modules.motion' options: {', '.join(unknown)}")

        values = {key: value for key, value in options.items() if key != "enabled"}
        values["zones"] = parse_zones(values.get("zones"))
        settings = cls(**values)
        settings._validate()
        return settings

    def _validate(self) -> None:
        if self.method not in METHODS:
            raise _error("method", f"one of {', '.join(METHODS)}", self.method)
        _check_int("threshold", self.threshold, 0, 255)
        _check_odd("blur", self.blur)
        _check_int("history", self.history, 1)
        _check_odd("kernel_size", self.kernel_size)
        _check_int("min_area", self.min_area, 0)
        _check_int("warmup_frames", self.warmup_frames, 0)
        _check_int("start_frames", self.start_frames, 1)
        _check_int("stop_frames", self.stop_frames, 1)
        if not _is_number(self.var_threshold) or self.var_threshold <= 0:
            raise _error("var_threshold", "a positive number", self.var_threshold)
        if not _is_number(self.background_ratio) or not 0 < self.background_ratio <= 1:
            raise _error("background_ratio", "a number above 0 and up to 1", self.background_ratio)
        if not isinstance(self.detect_shadows, bool):
            raise _error("detect_shadows", "true or false", self.detect_shadows)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _check_int(name: str, value: Any, minimum: int, maximum: int | None = None) -> None:
    valid = isinstance(value, int) and not isinstance(value, bool) and value >= minimum
    if not valid or (maximum is not None and value > maximum):
        expected = f"an integer from {minimum}" + (f" to {maximum}" if maximum is not None else "")
        raise _error(name, expected, value)


def _check_odd(name: str, value: Any) -> None:
    # Blur and morphology kernels need a center pixel, so their size must be odd.
    _check_int(name, value, 1)
    if value % 2 == 0:
        raise _error(name, "an odd integer (1 turns it off)", value)


def _error(name: str, expected: str, value: Any) -> ConfigError:
    return ConfigError(f"'modules.motion.{name}' must be {expected}, got {value!r}")
