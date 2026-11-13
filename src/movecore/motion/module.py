"""The motion module: mask → boxes → events, on every frame."""

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from movecore.core.debounce import Debouncer
from movecore.core.events import Event
from movecore.core.module import FrameContext, Module, ModuleResult
from movecore.motion.background_sub import BackgroundSubtractor
from movecore.motion.frame_diff import FrameDifferencer
from movecore.motion.morphology import Box, clean_mask, find_boxes
from movecore.motion.settings import MotionSettings
from movecore.motion.zones import moving_pixels
from movecore.output.overlay import draw_label
from movecore.preprocess import blur, to_gray

GREEN = (0, 200, 0)
ORANGE = (0, 140, 255)
GRAY = (160, 160, 160)


class MotionModule(Module):
    """Finds moving regions and reports ``motion.*`` and ``zone.*`` events.

    ``result.data`` holds ``mask`` (the cleaned mask), ``boxes`` (list of ``Box``),
    ``moving`` (debounced, bool) and ``active_zones`` (zone names with motion).
    """

    name = "motion"

    def __init__(self, options: Mapping[str, Any]) -> None:
        self.settings = MotionSettings.from_options(options)

    def setup(self) -> None:
        s = self.settings
        if s.method == "diff":
            self._detector: FrameDifferencer | BackgroundSubtractor = FrameDifferencer(s.threshold)
        else:
            self._detector = BackgroundSubtractor(
                s.method, s.history, s.var_threshold, s.detect_shadows, s.background_ratio
            )
        self._frames_seen = 0
        self._motion = Debouncer(s.start_frames, s.stop_frames)
        self._motion_since = 0
        self._zones = {zone.name: Debouncer(s.start_frames, s.stop_frames) for zone in s.zones}
        self._zone_since = dict.fromkeys(self._zones, 0)

    def process(self, ctx: FrameContext) -> ModuleResult:
        s = self.settings
        # Frame differencing compares brightness only. MOG2/KNN get color, so their shadow
        # test can tell a dimmer version of the background from a different-colored object.
        image = to_gray(ctx.frame) if s.method == "diff" else ctx.frame
        mask = clean_mask(self._detector.apply(blur(image, s.blur)), s.kernel_size)
        boxes = find_boxes(mask, s.min_area)

        events: list[Event] = []
        self._frames_seen += 1
        # The background model starts empty and the camera may still be adjusting exposure,
        # so the first frames would report motion that isn't there.
        if self._frames_seen > s.warmup_frames:
            events += self._update_motion(boxes, ctx)
            events += self._update_zones(mask, ctx)

        active_zones = [name for name, zone in self._zones.items() if zone.active]
        data = {"mask": mask, "boxes": boxes, "moving": self._motion.active}
        data["active_zones"] = active_zones
        return ModuleResult(data=data, events=events)

    def draw(self, frame: np.ndarray, result: ModuleResult) -> None:
        frame_height, frame_width = frame.shape[:2]
        for zone in self.settings.zones:
            x, y, width, height = zone.to_pixels(frame_width, frame_height)
            color = ORANGE if zone.name in result.data["active_zones"] else GRAY
            cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), color, 2)
            # Name at the bottom: the top-left corner is where the FPS label goes.
            label_origin = (x + 6, y + height - 10)
            cv2.putText(frame, zone.name, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for box in result.data["boxes"]:
            corner = (box.x + box.width - 1, box.y + box.height - 1)
            cv2.rectangle(frame, (box.x, box.y), corner, GREEN, 2)
        if result.data["moving"]:
            draw_label(frame, "MOTION", origin=(10, 52))

    def _update_motion(self, boxes: list[Box], ctx: FrameContext) -> list[Event]:
        change = self._motion.update(bool(boxes))
        if change == "start":
            self._motion_since = ctx.timestamp_ms
            payload = {"boxes": [list(box[:4]) for box in boxes]}
            return [self._event("motion.start", ctx, payload)]
        if change == "stop":
            payload = {"duration_ms": ctx.timestamp_ms - self._motion_since}
            return [self._event("motion.stop", ctx, payload)]
        return []

    def _update_zones(self, mask: np.ndarray, ctx: FrameContext) -> list[Event]:
        frame_height, frame_width = mask.shape
        events = []
        for zone in self.settings.zones:
            # The same size rule as for boxes, so a zone's size doesn't change its sensitivity.
            moving = moving_pixels(mask, zone.to_pixels(frame_width, frame_height))
            change = self._zones[zone.name].update(moving >= self.settings.min_area)
            if change == "start":
                self._zone_since[zone.name] = ctx.timestamp_ms
                events.append(self._event("zone.enter", ctx, {"zone": zone.name}))
            elif change == "stop":
                duration = ctx.timestamp_ms - self._zone_since[zone.name]
                payload = {"zone": zone.name, "duration_ms": duration}
                events.append(self._event("zone.leave", ctx, payload))
        return events

    def _event(self, event_type: str, ctx: FrameContext, payload: dict[str, Any]) -> Event:
        return Event(event_type, self.name, ctx.index, ctx.timestamp_ms, payload)
