"""Named areas of the frame that report when motion enters and leaves them."""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from movecore.core.config import ConfigError

ZONE_FORMAT = "[x, y, width, height] as fractions of the frame, e.g. [0.0, 0.0, 0.5, 1.0]"


@dataclass(frozen=True)
class Zone:
    """A rectangle in fractions of the frame, so it fits any resolution."""

    name: str
    x: float
    y: float
    width: float
    height: float

    def to_pixels(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        """The zone as ``(x, y, width, height)`` in pixels."""
        left, top = round(self.x * frame_width), round(self.y * frame_height)
        right = round((self.x + self.width) * frame_width)
        bottom = round((self.y + self.height) * frame_height)
        return left, top, right - left, bottom - top


def parse_zones(section: Any) -> tuple[Zone, ...]:
    if section is None:
        return ()
    if not isinstance(section, dict):
        raise ConfigError(f"'modules.motion.zones' must map names to {ZONE_FORMAT}")

    zones = []
    for name, rect in section.items():
        numbers = isinstance(rect, list) and len(rect) == 4
        numbers = numbers and all(
            isinstance(value, int | float) and not isinstance(value, bool) for value in rect
        )
        if not numbers:
            raise ConfigError(f"Zone {name!r} must be {ZONE_FORMAT}")
        x, y, width, height = (float(value) for value in rect)
        # A small tolerance so [0.5, 0, 0.5, 1] isn't rejected for float rounding.
        inside = 0 <= x < 1 and 0 <= y < 1 and width > 0 and height > 0
        inside = inside and x + width <= 1 + 1e-9 and y + height <= 1 + 1e-9
        if not inside:
            raise ConfigError(f"Zone {name!r} must lie inside the frame: {ZONE_FORMAT}")
        zones.append(Zone(str(name), x, y, width, height))
    return tuple(zones)


def moving_pixels(mask: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    """How many pixels inside the zone are marked as moving."""
    x, y, width, height = rect
    return cv2.countNonZero(mask[y : y + height, x : x + width])
