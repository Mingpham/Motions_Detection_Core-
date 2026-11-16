"""Gestures made of movement over time, not of a pose: paths and swipes.

Both keep a point's recent positions in a sliding time window (``PathTracker``):

- **Palm path:** where the whole hand has been. Steady, because the palm only moves when the
  hand does. Kept for later uses (drawing in the air, following a hand across zones).
- **Swipe:** a quick flick of the index fingertip. Reported when, within a short window, the
  fingertip travelled far enough, fast, mostly in one direction, along a fairly straight
  path, with the index finger extended the whole time.

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
INDEX_TIP = FINGERS["index"][2]
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
    distance: float  # hand sizes from start to end
    duration_ms: int

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


def index_tip(hand: Hand, width: int, height: int) -> tuple[float, float]:
    x, y = hand.image[INDEX_TIP, :2] * (width, height)
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


class SwipeDetector:
    """Watches one hand's index fingertip and reports quick flicks.

    At most one swipe per ``cooldown_ms``; a second one in the same direction also waits
    until the fingertip has slowed down. A quick back-and-forth gives two swipes.
    """

    def __init__(self, window_ms: int, min_distance: float, cooldown_ms: int) -> None:
        self.min_distance = min_distance
        self.cooldown_ms = cooldown_ms
        self._path = PathTracker(window_ms)
        self._quiet_until = -1
        self._last_direction: str | None = None

    @property
    def trail(self) -> list[tuple[float, float]]:
        """The fingertip positions in the current window, oldest first, for drawing."""
        return self._path.trail

    def reset(self) -> None:
        """Forget the path, e.g. when the hand leaves: a new hand must not join an old one."""
        self._path.clear()
        self._last_direction = None

    def update(
        self, hand: Hand, index_extended: bool, timestamp_ms: int, width: int, height: int
    ) -> Swipe | None:
        # Only a pointing finger swipes. Without this, opening a fist would look like an "up"
        # swipe: the fingertip shoots upward as the finger straightens.
        if not index_extended:
            self._path.clear()
            return None
        self._path.add(
            timestamp_ms, index_tip(hand, width, height), hand_size_px(hand, width, height)
        )
        if self._slowed_down():
            self._last_direction = None
        if timestamp_ms < self._quiet_until or len(self._path) < 3:
            return None

        swipe = self._measure()
        if swipe is None or swipe.direction == self._last_direction:
            return None
        # Start fresh, so the end of this swipe isn't counted again as a new one.
        self._path.clear()
        self._quiet_until = timestamp_ms + self.cooldown_ms
        self._last_direction = swipe.direction
        return swipe

    def _slowed_down(self) -> bool:
        span_ms = self._path.span_ms
        if span_ms < MIN_SPAN_MS:
            return False
        speed = self._path.distance() / span_ms * 1000
        slowest_swipe = self.min_distance / self._path.window_ms * 1000
        return speed < REARM_SPEED_SHARE * slowest_swipe

    def _measure(self) -> Swipe | None:
        distance = self._path.distance()
        if distance < self.min_distance:
            return None
        if self._path.straightness() < MIN_STRAIGHTNESS:
            return None  # a zigzag or a circle, not a swipe
        dx, dy = self._path.displacement
        if abs(dx) >= DOMINANCE * abs(dy):
            direction = "right" if dx > 0 else "left"
        elif abs(dy) >= DOMINANCE * abs(dx):
            direction = "down" if dy > 0 else "up"  # image y grows downward
        else:
            return None  # diagonal
        return Swipe(direction, distance, self._path.span_ms)
