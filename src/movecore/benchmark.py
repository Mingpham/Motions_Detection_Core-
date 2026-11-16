"""Measure frames per second and peak memory for one config, headless.

Used by scripts/benchmark.py, which runs each config in its own process: peak memory only
ever grows within a process, so measuring several configs in one would mix them up.
"""

import os
import resource
import sys
import time
from dataclasses import asdict, dataclass

from movecore.app import build_modules
from movecore.capture import open_source
from movecore.core.config import AppConfig
from movecore.core.pipeline import Pipeline
from movecore.preprocess import preprocess

# The v1 requirements (CLAUDE.md, section 4).
TARGET_FPS = 20.0
TARGET_MEMORY_MB = 1500.0
# The first frames are slower (model start-up, caches, camera exposure); they're not timed.
WARMUP_FRAMES = 30


@dataclass(frozen=True)
class Measurement:
    frames: int  # timed frames, after the warm-up
    seconds: float
    peak_memory_mb: float

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0

    @property
    def ms_per_frame(self) -> float:
        return 1000 / self.fps if self.fps else float("inf")

    @property
    def meets_targets(self) -> bool:
        return self.fps >= TARGET_FPS and self.peak_memory_mb < TARGET_MEMORY_MB

    def to_dict(self) -> dict[str, float]:
        return asdict(self) | {"fps": self.fps}


def measure(config: AppConfig, source: str | None, frames: int) -> Measurement:
    """Process ``frames`` frames after the warm-up, the way ``movecore --headless`` does."""
    width, height = config.input["width"], config.input["height"]
    flip = config.input.get("flip", False)
    modules, _ = build_modules(config)
    timed, start = 0, 0.0
    with open_source(source, config.input, loop=True) as frame_source, Pipeline(modules) as pipe:
        for index, (raw, timestamp_ms) in enumerate(frame_source.frames()):
            if index == WARMUP_FRAMES:
                start = time.perf_counter()
            pipe.process_frame(preprocess(raw, width, height, flip), index, timestamp_ms)
            if index >= WARMUP_FRAMES:
                timed += 1
                if timed >= frames:
                    break
    seconds = time.perf_counter() - start if timed else 0.0
    return Measurement(timed, seconds, peak_memory_mb())


def peak_memory_mb() -> float:
    """The most memory this process has used so far (its peak resident set size)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def machine_load() -> tuple[float, int]:
    """The 1-minute load average and the number of CPU cores."""
    return os.getloadavg()[0], os.cpu_count() or 1


def busy_warning() -> str | None:
    """A warning when other programs keep the CPU busy enough to skew the numbers."""
    load, cores = machine_load()
    if load > cores / 2:
        return (
            f"The machine is busy (load {load:.1f} on {cores} cores): close other heavy apps "
            "(Docker, IDEs, browsers) for numbers that reflect MoveCore alone."
        )
    return None
