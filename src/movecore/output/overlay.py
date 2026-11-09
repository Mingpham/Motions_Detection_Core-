"""Drawing helpers for the annotated view."""

import time
from collections.abc import Callable

import cv2
import numpy as np

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class FpsCounter:
    """Frames per second, smoothed so the number on screen doesn't flicker.

    Each tick blends the newest measurement into the running value:
    ``fps = smoothing * fps + (1 - smoothing) * instant_fps``.
    """

    def __init__(
        self, smoothing: float = 0.9, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing
        self.fps = 0.0
        self._clock = clock
        self._last: float | None = None

    def tick(self) -> float:
        """Record that a frame finished and return the smoothed FPS."""
        now = self._clock()
        if self._last is not None and now > self._last:
            instant = 1 / (now - self._last)
            if self.fps == 0:
                self.fps = instant
            else:
                self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant
        self._last = now
        return self.fps


def draw_label(frame: np.ndarray, text: str, origin: tuple[int, int] = (10, 24)) -> None:
    """Draw white text on a black box, readable on any background. Modifies ``frame``."""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    top_left = (x - 4, y - text_height - 4)
    bottom_right = (x + text_width + 4, y + baseline + 4)
    cv2.rectangle(frame, top_left, bottom_right, BLACK, cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, scale, WHITE, thickness, cv2.LINE_AA)


def side_by_side(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The frame with a black-and-white mask of the same size next to it."""
    return np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
