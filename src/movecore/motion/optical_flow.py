"""Optical flow: how far, and in which direction, points moved between two frames."""

import math

import cv2
import numpy as np

# A 21×21 search window over 4 pyramid levels (full, ½, ¼, ⅛ size), so moves much larger
# than the window are still found on the smaller levels.
LK_PARAMS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
}
# Tracking a point forward then back should land where it started; if not, it was lost.
MAX_ROUND_TRIP_PX = 1.0


def sparse_flow(
    previous: np.ndarray, current: np.ndarray, max_corners: int, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Track corners from ``previous`` to ``current`` with Lucas-Kanade.

    Corners are only looked for where ``mask`` is non-zero, if given. Returns
    ``(points, moves)``: start positions and how far each moved, both ``(N, 2)`` in pixels.
    """
    empty = np.empty((0, 2), dtype=np.float32)
    corners = cv2.goodFeaturesToTrack(
        previous, maxCorners=max_corners, qualityLevel=0.01, minDistance=7, mask=mask
    )
    if corners is None:
        return empty, empty

    forward, found, _ = cv2.calcOpticalFlowPyrLK(previous, current, corners, None, **LK_PARAMS)
    back, found_back, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **LK_PARAMS)
    round_trip = np.linalg.norm(corners - back, axis=2).ravel()
    good = (found.ravel() == 1) & (found_back.ravel() == 1) & (round_trip < MAX_ROUND_TRIP_PX)

    points = corners.reshape(-1, 2)[good]
    return points, forward.reshape(-1, 2)[good] - points


def dense_flow(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Farneback flow: an ``(H, W, 2)`` array of every pixel's move ``(dx, dy)`` in pixels."""
    return cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        pyr_scale=0.5,  # each pyramid level is half the size of the one below
        levels=3,
        winsize=15,  # bigger is smoother and handles faster motion, but blurs edges
        iterations=3,
        poly_n=5,  # neighbourhood used to fit each pixel's polynomial
        poly_sigma=1.2,
        flags=0,
    )


def flow_to_color(velocities: np.ndarray, full_speed: float) -> np.ndarray:
    """Picture a flow field: hue = direction, brightness = speed (``full_speed`` is brightest)."""
    speed, angle = cv2.cartToPolar(velocities[..., 0], velocities[..., 1], angleInDegrees=True)
    hsv = np.empty((*velocities.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)  # OpenCV hue runs 0-179, half of 360 degrees
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(speed / full_speed * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def direction_name(vx: float, vy: float) -> str:
    """'left', 'right', 'up' or 'down', whichever axis dominates. Image y grows downwards."""
    if abs(vx) >= abs(vy):
        return "right" if vx > 0 else "left"
    return "down" if vy > 0 else "up"


def screen_angle(vx: float, vy: float) -> float:
    """Degrees counter-clockwise from "right", as seen on screen: 0 right, 90 up."""
    return math.degrees(math.atan2(-vy, vx)) % 360
