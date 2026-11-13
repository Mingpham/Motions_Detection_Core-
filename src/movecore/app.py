"""The run loop: read frames, prepare them, run the modules, show the result."""

import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np

from movecore.capture import open_source
from movecore.core.config import AppConfig
from movecore.core.events import ALL_EVENTS, Event, EventBus
from movecore.core.module import FrameContext, Module
from movecore.core.pipeline import Pipeline
from movecore.hands import HandsModule
from movecore.motion import MotionModule, OpticalFlowModule
from movecore.output import Display, EventLogger, FpsCounter, draw_label, side_by_side
from movecore.preprocess import preprocess

ModuleFactory = Callable[[dict[str, Any]], Module]

# Module name in the config -> class. Each milestone registers its modules here.
# Modules run in this order, so a module comes after the ones whose results it reads.
MODULES: dict[str, ModuleFactory] = {
    "motion": MotionModule,
    "optical_flow": OpticalFlowModule,  # reads motion's mask and boxes when present
    "hands": HandsModule,
}


@dataclass(frozen=True)
class RunSummary:
    frames: int
    seconds: float
    events: int = 0

    @property
    def average_fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


def build_modules(config: AppConfig) -> tuple[list[Module], list[str]]:
    """Create the enabled modules that exist so far; also return the names still missing."""
    enabled = config.enabled_modules()
    # Registry order, not config order, so readers always run after what they read.
    modules = [build(config.modules[name]) for name, build in MODULES.items() if name in enabled]
    missing = [name for name in enabled if name not in MODULES]
    return modules, missing


def print_event(event: Event) -> None:
    details = " ".join(f"{key}={value}" for key, value in event.payload.items() if key != "boxes")
    print(f"[{event.timestamp_ms / 1000:8.2f}s] {event.type:<13} {details}".rstrip())


def find_mask(ctx: FrameContext) -> np.ndarray | None:
    """The first mask any module produced for this frame, or None."""
    return next((r.data["mask"] for r in ctx.results.values() if "mask" in r.data), None)


def run(
    config: AppConfig,
    source: str | None = None,
    *,
    headless: bool = False,
    loop: bool = False,
    max_frames: int | None = None,
    show_mask: bool = False,
) -> RunSummary:
    width, height = config.input["width"], config.input["height"]
    flip = config.input.get("flip", False)
    overlay = config.output.get("overlay", True)
    show_fps = overlay and config.output.get("show_fps", True)
    event_log = config.output.get("event_log")

    modules, missing = build_modules(config)
    if missing:
        print(f"Not built yet, skipped: {', '.join(missing)}")

    bus = EventBus()
    events: list[Event] = []
    bus.subscribe(ALL_EVENTS, events.append)
    bus.subscribe(ALL_EVENTS, print_event)

    fps_counter = FpsCounter()
    frames = 0
    start = time.perf_counter()
    # Entered left to right, so the log and window only open once the source has opened.
    with (
        open_source(source, config.input, loop=loop) as frame_source,
        EventLogger(event_log) if event_log else nullcontext() as logger,
        Pipeline(modules, bus) as pipeline,
        nullcontext() if headless else Display() as display,
    ):
        print(f"Source:  {frame_source.description}")
        if logger is not None:
            print(f"Events:  {logger.path}")
            bus.subscribe(ALL_EVENTS, logger.write)

        for index, (raw, timestamp_ms) in enumerate(frame_source.frames()):
            if index == 0:
                raw_height, raw_width = raw.shape[:2]
                print(f"Frames:  {raw_width}x{raw_height} in, {width}x{height} processed")

            frame = preprocess(raw, width, height, flip)
            ctx = pipeline.process_frame(frame, index, timestamp_ms)
            frames += 1
            fps = fps_counter.tick()

            if display is not None:
                view = ctx.frame.copy()
                if overlay:
                    for module in modules:
                        module.draw(view, ctx.results[module.name])
                if show_fps:
                    draw_label(view, f"FPS {fps:.1f}")
                mask = find_mask(ctx) if show_mask else None
                if not display.show(view if mask is None else side_by_side(view, mask)):
                    break
            if max_frames is not None and frames >= max_frames:
                break
    return RunSummary(frames, time.perf_counter() - start, len(events))
