"""Gestures from plain geometry on the 21 landmarks: no training, every rule readable.

All rules take ``world`` landmarks (metres, shape (21, 3)), so they work whatever the hand's
distance from the camera, and mostly whatever its rotation, because they compare distances
rather than "is the tip above the knuckle".
"""

import numpy as np

from movecore.hands.landmarks import (
    FINGERS,
    MIDDLE_MCP,
    PINKY_MCP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
)

# A finger is extended when its tip is at least this much farther from the reference point
# than its middle joint is. Measured on sample photos: extended fingers 1.19-1.61, curled
# ones 0.60-0.99.
EXTENDED_RATIO = 1.1
INDEX_TIP = FINGERS["index"][2]


def distance(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a] - points[b]))


def hand_size(points: np.ndarray) -> float:
    """Wrist to middle knuckle: a length that doesn't change when the fingers move."""
    return distance(points, WRIST, MIDDLE_MCP)


def extended_fingers(points: np.ndarray) -> tuple[bool, ...]:
    """Which fingers are straight, in order thumb, index, middle, ring, pinky.

    A straight finger reaches far from the wrist; a curled one folds its tip back in, closer
    than its own middle joint. The thumb bends sideways across the palm, so it's measured
    from the pinky knuckle instead of the wrist.
    """
    result = []
    for name, (_, joint, tip) in FINGERS.items():
        origin = PINKY_MCP if name == "thumb" else WRIST
        reach = distance(points, origin, tip) / distance(points, origin, joint)
        result.append(reach > EXTENDED_RATIO)
    return tuple(result)


def is_pinch(points: np.ndarray, pinch_ratio: float) -> bool:
    """Thumb tip and index tip closer than ``pinch_ratio`` hand sizes."""
    return distance(points, THUMB_TIP, INDEX_TIP) < pinch_ratio * hand_size(points)


def thumb_direction(points: np.ndarray) -> str | None:
    """'up' or 'down' when the thumb points within 45° of straight up or down, else None.

    World y points down, like image y, so up is negative.
    """
    dx, dy = points[THUMB_TIP, :2] - points[THUMB_MCP, :2]
    if abs(dy) <= abs(dx):
        return None
    return "up" if dy < 0 else "down"


def rule_gesture(points: np.ndarray, pinch_ratio: float) -> str:
    """One label per hand: 'pinch', 'thumb_up', 'thumb_down', or 'fingers_<count>'.

    Pinch is checked first because a pinching hand can have any fingers up.
    """
    if is_pinch(points, pinch_ratio):
        return "pinch"
    fingers = extended_fingers(points)
    if fingers == (True, False, False, False, False):
        direction = thumb_direction(points)
        if direction is not None:
            return f"thumb_{direction}"
    return f"fingers_{sum(fingers)}"
