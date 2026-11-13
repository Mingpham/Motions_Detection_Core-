"""The 21 hand landmarks: their numbering, the bones between them, and one detected hand."""

from dataclasses import dataclass

import numpy as np

WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
MIDDLE_MCP = 9
PINKY_MCP = 17

# Each finger as (knuckle, middle joint, fingertip). The thumb's "middle joint" is its IP
# joint; the others use the PIP joint, the one in the middle of the finger.
FINGERS: dict[str, tuple[int, int, int]] = {
    "thumb": (2, 3, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}
TIPS = tuple(tip for _, _, tip in FINGERS.values())

# The bones to draw: each finger from the wrist out, then across the palm.
CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)  # fmt: skip


@dataclass(frozen=True)
class Hand:
    """One hand found in one frame.

    ``image`` holds the 21 points as ``(x, y, z)``: x and y are fractions of the frame's width
    and height, z is depth relative to the wrist. ``world`` holds the same points in metres
    around the hand's center, so distances and angles don't depend on how far away the hand
    is or on the frame's shape. Gesture rules use ``world``; drawing uses ``image``.
    """

    handedness: str  # "Left" or "Right", from the person's point of view
    score: float  # how sure the model is about ``handedness``
    image: np.ndarray  # (21, 3)
    world: np.ndarray  # (21, 3)
    gesture: str | None = None  # built-in recognizer's label, when it ran and was sure
    gesture_score: float = 0.0

    def pixels(self, width: int, height: int) -> np.ndarray:
        """The points as integer ``(x, y)`` pixel positions, shape (21, 2)."""
        return np.rint(self.image[:, :2] * (width, height)).astype(int)
