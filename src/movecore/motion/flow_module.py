"""The optical flow module: direction and speed of what's moving."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import cv2
import numpy as np

from movecore.core.events import Event
from movecore.core.module import FrameContext, Module, ModuleResult
from movecore.core.options import OptionChecks
from movecore.motion.debounce import StableLabel
from movecore.motion.morphology import Box
from movecore.motion.optical_flow import (
    dense_flow,
    direction_name,
    flow_to_color,
    screen_angle,
    sparse_flow,
)
from movecore.output.overlay import draw_label
from movecore.preprocess import to_gray

MODES = ("sparse", "dense")
CHECKS = OptionChecks("modules.optical_flow")
# Dense: a direction is only reported when at least this share of the frame is moving.
DENSE_MIN_MOVING_SHARE = 0.002
# Dense view: this speed (px/s) is drawn at full brightness.
VIEW_FULL_SPEED = 600.0
# Arrows show where a point will be this many seconds from now, so they're long enough to see.
ARROW_SECONDS = 0.25
YELLOW = (0, 220, 255)
CYAN = (255, 220, 0)


@dataclass(frozen=True)
class FlowSettings:
    mode: str = "sparse"
    max_corners: int = 200
    min_points: int = 5
    min_speed: float = 40.0
    dense_scale: float = 0.5
    stable_frames: int = 3

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "FlowSettings":
        CHECKS.reject_unknown(options, {field.name for field in fields(cls)} | {"enabled"})
        settings = cls(**{key: value for key, value in options.items() if key != "enabled"})
        CHECKS.choice("mode", settings.mode, MODES)
        CHECKS.integer("max_corners", settings.max_corners, 1)
        CHECKS.integer("min_points", settings.min_points, 1)
        CHECKS.number("min_speed", settings.min_speed)
        CHECKS.number("dense_scale", settings.dense_scale, at_most=1)
        CHECKS.integer("stable_frames", settings.stable_frames, 1)
        return settings


@dataclass
class _Previous:
    gray: np.ndarray
    mask: np.ndarray | None
    timestamp_ms: int


class OpticalFlowModule(Module):
    """Measures how things move, in pixels per second, and reports ``flow.direction`` events.

    ``result.data`` holds ``velocity`` ``(vx, vy)``, ``speed``, ``direction`` ('left', 'right',
    'up', 'down' or None), ``angle_deg`` (or None) and ``boxes``: ``(Box, vx, vy)`` for each
    motion box when the motion module runs first. Sparse mode adds ``points`` and ``moves``;
    dense mode adds ``view``, a color picture of the flow.
    """

    name = "optical_flow"

    def __init__(self, options: Mapping[str, Any]) -> None:
        self.settings = FlowSettings.from_options(options)

    def setup(self) -> None:
        self._previous: _Previous | None = None
        self._direction = StableLabel(self.settings.stable_frames)

    def process(self, ctx: FrameContext) -> ModuleResult:
        motion = ctx.results.get("motion")
        mask = motion.data.get("mask") if motion else None
        boxes: list[Box] = motion.data.get("boxes", []) if motion else []

        gray = to_gray(ctx.frame)
        previous, self._previous = self._previous, _Previous(gray, mask, ctx.timestamp_ms)
        if previous is None:
            return ModuleResult(data=_still())

        seconds = (ctx.timestamp_ms - previous.timestamp_ms) / 1000
        if self.settings.mode == "sparse":
            data = self._sparse(previous, gray, seconds, boxes)
        else:
            data = self._dense(previous, gray, seconds, boxes, ctx.frame.shape)

        events = []
        if self._direction.update(data["direction"]) and self._direction.current is not None:
            payload = {
                "direction": data["direction"],
                "speed": round(data["speed"]),
                "angle_deg": round(data["angle_deg"]) % 360,  # 359.7 rounds to 360, which is 0
            }
            events.append(Event("flow.direction", self.name, ctx.index, ctx.timestamp_ms, payload))
        return ModuleResult(data=data, events=events)

    def _sparse(
        self, previous: _Previous, gray: np.ndarray, seconds: float, boxes: list[Box]
    ) -> dict[str, Any]:
        # Only look for corners where the motion module saw movement: still background
        # corners would just add zeros.
        points, moves = sparse_flow(previous.gray, gray, self.settings.max_corners, previous.mask)
        velocities = moves / seconds
        moving = np.linalg.norm(velocities, axis=1) >= self.settings.min_speed
        enough = moving.sum() >= self.settings.min_points
        data = _summary(velocities[moving] if enough else None)
        data["points"], data["moves"] = points[moving], moves[moving]
        data["boxes"] = [
            (box, *_median(velocities[moving & _inside(points, box)])) for box in boxes
        ]
        return data

    def _dense(
        self,
        previous: _Previous,
        gray: np.ndarray,
        seconds: float,
        boxes: list[Box],
        frame_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        scale = self.settings.dense_scale
        # Flow for every pixel is heavy, so it runs on a smaller copy of both frames.
        flow = dense_flow(_shrink(previous.gray, scale), _shrink(gray, scale))
        velocities = flow / scale / seconds  # back to full-size pixels per second
        moving = np.linalg.norm(velocities, axis=2) >= self.settings.min_speed
        enough = moving.mean() >= DENSE_MIN_MOVING_SHARE
        data = _summary(velocities[moving] if enough else None)

        height, width = frame_shape[:2]
        view = flow_to_color(velocities, VIEW_FULL_SPEED)
        data["view"] = cv2.resize(view, (width, height), interpolation=cv2.INTER_NEAREST)
        data["boxes"] = []
        for box in boxes:
            x0, y0 = int(box.x * scale), int(box.y * scale)
            x1, y1 = int((box.x + box.width) * scale), int((box.y + box.height) * scale)
            region, region_moving = velocities[y0:y1, x0:x1], moving[y0:y1, x0:x1]
            data["boxes"].append((box, *_median(region[region_moving])))
        return data

    def draw(self, frame: np.ndarray, result: ModuleResult) -> None:
        data = result.data
        if "view" in data:
            # Tint only the moving pixels, so the still parts of the image stay visible.
            moving = data["view"].any(axis=2)
            blended = cv2.addWeighted(frame, 0.4, data["view"], 0.6, 0)
            frame[moving] = blended[moving]
        for (x, y), (dx, dy) in zip(data.get("points", []), data.get("moves", []), strict=True):
            end = (round(x + dx * 3), round(y + dy * 3))  # 3× longer, so short moves show
            cv2.arrowedLine(frame, (round(x), round(y)), end, YELLOW, 1, tipLength=0.3)
        for box, vx, vy in data["boxes"]:
            if vx or vy:
                center = (box.x + box.width // 2, box.y + box.height // 2)
                end = (round(center[0] + vx * ARROW_SECONDS), round(center[1] + vy * ARROW_SECONDS))
                cv2.arrowedLine(frame, center, end, CYAN, 3, tipLength=0.25)
        if data["direction"]:
            draw_label(frame, f"{data['direction']} {data['speed']:.0f} px/s", origin=(10, 80))


def _still() -> dict[str, Any]:
    return {**_summary(None), "boxes": []}


def _summary(velocities: np.ndarray | None) -> dict[str, Any]:
    """Overall motion from the moving points' velocities, or "still" when there aren't enough."""
    if velocities is None or len(velocities) == 0:
        return {"velocity": (0.0, 0.0), "speed": 0.0, "direction": None, "angle_deg": None}
    # The median ignores a few badly tracked points that the mean would follow.
    vx, vy = _median(velocities)
    return {
        "velocity": (vx, vy),
        "speed": float(np.hypot(vx, vy)),
        "direction": direction_name(vx, vy),
        "angle_deg": screen_angle(vx, vy),
    }


def _median(velocities: np.ndarray) -> tuple[float, float]:
    if len(velocities) == 0:
        return 0.0, 0.0
    vx, vy = np.median(velocities, axis=0)
    return float(vx), float(vy)


def _inside(points: np.ndarray, box: Box) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    return (x >= box.x) & (x < box.x + box.width) & (y >= box.y) & (y < box.y + box.height)


def _shrink(gray: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1:
        return gray
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
