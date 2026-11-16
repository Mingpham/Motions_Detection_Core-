"""Gestures made of movement over time, not of a pose: paths and swipes.

Both keep a point's recent positions in a sliding time window (``PathTracker``):

- **Palm path:** where the whole hand has been. Steady, because the palm only moves when the
  hand does. Kept for later uses (drawing in the air, following a hand across zones).
- **Swipe:** a quick flick of one or more fingertips. Each finger has its own path, kept
  only while that finger is extended. A swipe is reported when, within a short window, a
  fingertip travelled far enough, fast, mostly in one direction, along a fairly straight
  path. Fingers moving the same way at that moment are reported together (a whole-hand
  wave is one swipe with every extended finger, not five).

Distances are in hand sizes rather than pixels, so a small hand far from the camera swipes
the same as a big one close to it.
"""

from collections import deque
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from movecore.hands.landmarks import FINGERS, MIDDLE_MCP, WRIST, Hand

# The palm center: wrist and the four finger knuckles.
PALM = (0, 5, 9, 13, 17)
FINGER_NAMES = tuple(FINGERS)  # thumb, index, middle, ring, pinky
# The main direction must be at least this many times the other one (2 = within ~27°).
DOMINANCE = 2.0
# Distance from start to end, divided by the length of the path walked: 1 = a straight line.
MIN_STRAIGHTNESS = 0.7
# After a swipe, another in the same direction needs the fingertip to slow down first, below
# this share of the slowest speed that still makes a swipe. Without it, one long movement
# that outlasts the cooldown would count twice.
REARM_SPEED_SHARE = 1 / 3
# Speed is only judged on at least this much history.
MIN_SPAN_MS = 60


@dataclass(frozen=True)
class Swipe:
    direction: str  # 'left', 'right', 'up' or 'down', as seen on screen
    distance: float  # hand sizes from start to end, for the finger that moved most
    duration_ms: int
    fingers: tuple[str, ...] = ()  # every finger that swiped this way, thumb first

    @property
    def speed(self) -> float:
        """Hand sizes per second."""
        return self.distance / max(self.duration_ms, 1) * 1000


@dataclass(frozen=True)
class _Sample:
    timestamp_ms: int
    x: float
    y: float
    size: float  # hand size in pixels at that moment


def palm_center(hand: Hand, width: int, height: int) -> tuple[float, float]:
    points = hand.image[PALM, :2] * (width, height)
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def fingertip(hand: Hand, finger: str, width: int, height: int) -> tuple[float, float]:
    x, y = hand.image[FINGERS[finger][2], :2] * (width, height)
    return float(x), float(y)


def hand_size_px(hand: Hand, width: int, height: int) -> float:
    wrist, knuckle = hand.image[[WRIST, MIDDLE_MCP], :2] * (width, height)
    return float(np.linalg.norm(knuckle - wrist))


class PathTracker:
    """One point's positions over the last ``window_ms``, with distances in hand sizes."""

    def __init__(self, window_ms: int) -> None:
        self.window_ms = window_ms
        self._samples: deque[_Sample] = deque()

    def add(self, timestamp_ms: int, point: tuple[float, float], size: float) -> None:
        self._samples.append(_Sample(timestamp_ms, *point, size))
        while timestamp_ms - self._samples[0].timestamp_ms > self.window_ms:
            self._samples.popleft()

    def clear(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def trail(self) -> list[tuple[float, float]]:
        """The positions in the window, oldest first."""
        return [(s.x, s.y) for s in self._samples]

    @property
    def span_ms(self) -> int:
        if not self._samples:
            return 0
        return self._samples[-1].timestamp_ms - self._samples[0].timestamp_ms

    @property
    def displacement(self) -> tuple[float, float]:
        """Start to end of the window, in pixels."""
        if not self._samples:
            return 0.0, 0.0
        first, last = self._samples[0], self._samples[-1]
        return last.x - first.x, last.y - first.y

    def distance(self) -> float:
        """Start to end of the window, in hand sizes."""
        size = float(np.median([s.size for s in self._samples])) if self._samples else 0.0
        if size <= 0:
            return 0.0
        return float(np.hypot(*self.displacement)) / size

    def straightness(self) -> float:
        """Start-to-end distance over the length of the path walked: 1 = a straight line."""
        walked = sum(float(np.hypot(b.x - a.x, b.y - a.y)) for a, b in pairwise(self._samples))
        return float(np.hypot(*self.displacement)) / walked if walked > 0 else 0.0


def measure_swipe(path: PathTracker, min_distance: float) -> Swipe | None:
    """The swipe this path makes right now, if it passes the distance, straightness and
    direction tests."""
    distance = path.distance()
    if distance < min_distance:
        return None
    if path.straightness() < MIN_STRAIGHTNESS:
        return None  # a zigzag or a circle, not a swipe
    dx, dy = path.displacement
    if abs(dx) >= DOMINANCE * abs(dy):
        direction = "right" if dx > 0 else "left"
    elif abs(dy) >= DOMINANCE * abs(dx):
        direction = "down" if dy > 0 else "up"  # image y grows downward
    else:
        return None  # diagonal
    return Swipe(direction, distance, path.span_ms)


class SwipeDetector:
    """Watches one hand's fingertips and reports quick flicks.

    Each finger in ``fingers`` has its own path. At most one swipe per hand per
    ``cooldown_ms``; a second one in the same direction also waits until the fingertips have
    slowed down. A quick back-and-forth gives two swipes.
    """

    def __init__(
        self,
        window_ms: int,
        min_distance: float,
        cooldown_ms: int,
        fingers: tuple[str, ...] = FINGER_NAMES,
    ) -> None:
        self.window_ms = window_ms
        self.min_distance = min_distance
        self.cooldown_ms = cooldown_ms
        self._paths = {finger: PathTracker(window_ms) for finger in fingers}
        self._quiet_until = -1
        self._last_direction: str | None = None

    @property
    def trails(self) -> dict[str, list[tuple[float, float]]]:
        """Each watched fingertip's positions in the current window, oldest first."""
        return {finger: path.trail for finger, path in self._paths.items()}

    def reset(self) -> None:
        """Forget the paths, e.g. when the hand leaves: a new hand must not join an old one."""
        self._clear()
        self._last_direction = None

    def update(
        self, hand: Hand, extended: tuple[bool, ...], timestamp_ms: int, width: int, height: int
    ) -> Swipe | None:
        """Add this frame's fingertips; ``extended`` is per finger, thumb to pinky."""
        size = hand_size_px(hand, width, height)
        for finger, path in self._paths.items():
            # Only an extended finger swipes. Without this, opening a fist would look like an
            # "up" swipe: the fingertips shoot upward as the fingers straighten.
            if extended[FINGER_NAMES.index(finger)]:
                path.add(timestamp_ms, fingertip(hand, finger, width, height), size)
            else:
                path.clear()
        if self._slowed_down():
            self._last_direction = None
        if timestamp_ms < self._quiet_until:
            return None

        candidates = {
            finger: swipe
            for finger, path in self._paths.items()
            if len(path) >= 3 and (swipe := measure_swipe(path, self.min_distance)) is not None
        }
        if not candidates:
            return None
        leader = max(candidates.values(), key=lambda swipe: swipe.distance)
        if leader.direction == self._last_direction:
            return None
        # Fingers flicking the other way at the same moment are left out.
        fingers = tuple(
            finger
            for finger, swipe in sorted(candidates.items(), key=lambda c: FINGER_NAMES.index(c[0]))
            if swipe.direction == leader.direction
        )
        # Start fresh, so the end of this swipe isn't counted again as a new one.
        self._clear()
        self._quiet_until = timestamp_ms + self.cooldown_ms
        self._last_direction = leader.direction
        return Swipe(leader.direction, leader.distance, leader.duration_ms, fingers)

    def _clear(self) -> None:
        for path in self._paths.values():
            path.clear()

    def _slowed_down(self) -> bool:
        """True when the fingertips have been watched long enough and none moves fast.

        Too little history (e.g. right after a swipe cleared the paths) counts as "not yet":
        otherwise the rest of one long movement would re-arm at once and count again.
        """
        slowest_swipe = self.min_distance / self.window_ms * 1000
        judged = [path for path in self._paths.values() if path.span_ms >= MIN_SPAN_MS]
        return bool(judged) and all(
            path.distance() / path.span_ms * 1000 < REARM_SPEED_SHARE * slowest_swipe
            for path in judged
        )
