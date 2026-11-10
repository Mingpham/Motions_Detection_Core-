"""Checks for module options, so every module reports bad config values the same way."""

from collections.abc import Collection, Mapping
from typing import Any

from movecore.core.config import ConfigError


def is_number(value: Any) -> bool:
    # bool is a subclass of int, so `true` would otherwise pass as 1.
    return isinstance(value, int | float) and not isinstance(value, bool)


class OptionChecks:
    """Validation for one config section, such as ``modules.motion``."""

    def __init__(self, section: str) -> None:
        self.section = section

    def error(self, name: str, expected: str, value: Any) -> ConfigError:
        return ConfigError(f"'{self.section}.{name}' must be {expected}, got {value!r}")

    def reject_unknown(self, options: Mapping[str, Any], known: Collection[str]) -> None:
        # Catches typos like `min_aera`, which would otherwise silently do nothing.
        unknown = sorted(set(options) - set(known))
        if unknown:
            raise ConfigError(f"Unknown '{self.section}' options: {', '.join(unknown)}")

    def integer(self, name: str, value: Any, minimum: int, maximum: int | None = None) -> None:
        valid = isinstance(value, int) and not isinstance(value, bool) and value >= minimum
        if not valid or (maximum is not None and value > maximum):
            limit = f" to {maximum}" if maximum is not None else ""
            raise self.error(name, f"an integer from {minimum}{limit}", value)

    def odd(self, name: str, value: Any) -> None:
        # Blur and morphology kernels need a center pixel, so their size must be odd.
        self.integer(name, value, 1)
        if value % 2 == 0:
            raise self.error(name, "an odd integer (1 turns it off)", value)

    def number(self, name: str, value: Any, at_most: float | None = None) -> None:
        """A number above 0, and optionally no more than ``at_most``."""
        if not is_number(value) or value <= 0 or (at_most is not None and value > at_most):
            expected = "a positive number" if at_most is None else f"above 0 and up to {at_most}"
            raise self.error(name, expected, value)

    def choice(self, name: str, value: Any, choices: Collection[str]) -> None:
        if value not in choices:
            raise self.error(name, f"one of {', '.join(choices)}", value)

    def boolean(self, name: str, value: Any) -> None:
        if not isinstance(value, bool):
            raise self.error(name, "true or false", value)
