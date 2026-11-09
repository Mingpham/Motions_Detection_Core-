"""Background subtraction: learn what the empty scene looks like, flag what doesn't fit."""

import cv2
import numpy as np

# Background subtractors mark likely shadows with this value instead of 255.
SHADOW_VALUE = 127


class BackgroundSubtractor:
    """MOG2 or KNN background model that returns a foreground mask without shadows."""

    def __init__(
        self,
        method: str = "mog2",
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        background_ratio: float = 0.9,
    ) -> None:
        if method == "mog2":
            self._model = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=var_threshold, detectShadows=detect_shadows
            )
            # How much of a pixel's recent history counts as background. Lower keeps
            # still objects visible longer but flags flickering things more often.
            self._model.setBackgroundRatio(background_ratio)
        elif method == "knn":
            self._model = cv2.createBackgroundSubtractorKNN(
                history=history, detectShadows=detect_shadows
            )
        else:
            raise ValueError(f"Unknown background subtraction method: {method!r}")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Update the model with a gray or BGR frame and return a mask: 255 = foreground."""
        mask = self._model.apply(frame)
        # Keep only definite foreground; shadows (127) would make boxes grow along the floor.
        _, mask = cv2.threshold(mask, SHADOW_VALUE, 255, cv2.THRESH_BINARY)
        return mask
