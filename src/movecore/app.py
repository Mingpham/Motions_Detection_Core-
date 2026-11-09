"""The run loop: read frames, prepare them, run the modules, show the result."""

import time
from contextlib import nullcontext
from dataclasses import dataclass

from movecore.capture import open_source
from movecore.core.config import AppConfig
from movecore.core.module import Module
from movecore.core.pipeline import Pipeline
from movecore.output import Display, FpsCounter, draw_label
from movecore.preprocess import preprocess

# Module name in the config -> class. Each milestone registers its modules here.
MODULES: dict[str, type[Module]] = {}


@dataclass(frozen=True)
class RunSummary:
    frames: int
    seconds: float

    @property
    def average_fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


def build_modules(config: AppConfig) -> tuple[list[Module], list[str]]:
    """Create the enabled modules that exist so far; also return the names still missing."""
    enabled = config.enabled_modules()
    modules = [MODULES[name]() for name in enabled if name in MODULES]
    missing = [name for name in enabled if name not in MODULES]
    return modules, missing


def run(
    config: AppConfig,
    source: str | None = None,
    *,
    headless: bool = False,
    loop: bool = False,
    max_frames: int | None = None,
) -> RunSummary:
    width, height = config.input["width"], config.input["height"]
    flip = config.input.get("flip", False)
    show_fps = config.output.get("overlay", True) and config.output.get("show_fps", True)

    modules, missing = build_modules(config)
    if missing:
        print(f"Not built yet, skipped: {', '.join(missing)}")

    fps_counter = FpsCounter()
    frames = 0
    start = time.perf_counter()
    # Entered left to right, so the window only opens once the source has opened.
    with (
        open_source(source, config.input, loop=loop) as frame_source,
        Pipeline(modules) as pipeline,
        nullcontext() if headless else Display() as display,
    ):
        print(f"Source:  {frame_source.description}")
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
                if show_fps:
                    draw_label(view, f"FPS {fps:.1f}")
                if not display.show(view):
                    break
            if max_frames is not None and frames >= max_frames:
                break
    return RunSummary(frames, time.perf_counter() - start)
