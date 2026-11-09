"""Frame differencing: compare each frame with the one before it."""

import cv2
import numpy as np


class FrameDifferencer:
    """Marks pixels whose brightness changed by more than ``threshold`` since the last frame."""

    def __init__(self, threshold: int = 25) -> None:
        self.threshold = threshold
        self._previous: np.ndarray | None = None

    def apply(self, gray: np.ndarray) -> np.ndarray:
        """Return a mask: 255 where the pixel changed, 0 elsewhere."""
        previous, self._previous = self._previous, gray
        if previous is None or previous.shape != gray.shape:
            return np.zeros_like(gray)
        # absdiff, not `gray - previous`: uint8 subtraction wraps around (10 - 20 = 246).
        difference = cv2.absdiff(gray, previous)
        _, mask = cv2.threshold(difference, self.threshold, 255, cv2.THRESH_BINARY)
        return mask
