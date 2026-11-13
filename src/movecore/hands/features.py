"""Turn one hand's landmarks into the numbers a classifier learns from.

A classifier only sees numbers, so anything that changes them without changing the gesture
makes learning harder. Three such things are removed here:

- where the hand is: points are measured from the wrist;
- how big it looks: they're divided by the hand's size;
- which hand it is: a left-hand shape is the mirror image of a right-hand one, so its x is
  flipped. What counts is the shape in the image, not the real hand: in a mirrored camera
  view your right hand has a left-hand shape (``Hand.looks_like``).

The hand's rotation is kept on purpose: thumb up and thumb down differ only by rotation.
"""

import numpy as np

from movecore.hands.gesture_rules import hand_size
from movecore.hands.landmarks import WRIST

FEATURE_COUNT = 63  # 21 points × (x, y, z)


def hand_features(world: np.ndarray, looks_like: str) -> np.ndarray:
    """63 numbers for one hand, from its world landmarks (shape (21, 3), metres).

    ``looks_like`` is the side the hand's shape has in the image ("Left" or "Right").
    """
    points = world - world[WRIST]
    if looks_like == "Left":
        points = points * (-1, 1, 1)
    return (points / hand_size(world)).reshape(FEATURE_COUNT).astype(np.float32)
