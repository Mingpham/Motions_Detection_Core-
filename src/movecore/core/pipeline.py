"""The frame loop: run each module in order and publish what it finds."""

from collections.abc import Iterable, Sequence
from types import TracebackType
from typing import Self

import numpy as np

from movecore.core.events import EventBus
from movecore.core.module import FrameContext, Module


class Pipeline:
    """Runs modules in the given order, so detectors must come before their analyzers."""

    def __init__(self, modules: Sequence[Module], bus: EventBus | None = None) -> None:
        names = [module.name for module in modules]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate module names: {duplicates}")
        self.modules = list(modules)
        self.bus = bus or EventBus()
        self._started: list[Module] = []

    def __enter__(self) -> Self:
        try:
            for module in self.modules:
                module.setup()
                self._started.append(module)
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close started modules in reverse order, even if one of them fails."""
        errors: list[Exception] = []
        while self._started:
            try:
                self._started.pop().close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("Failed to close modules", errors)

    def process_frame(self, frame: np.ndarray, index: int, timestamp_ms: int) -> FrameContext:
        ctx = FrameContext(frame=frame, index=index, timestamp_ms=timestamp_ms)
        for module in self.modules:
            result = module.process(ctx)
            ctx.results[module.name] = result
            for event in result.events:
                self.bus.publish(event)
        return ctx

    def run(self, frames: Iterable[tuple[np.ndarray, int]]) -> int:
        """Process ``(frame, timestamp_ms)`` pairs and return how many were processed."""
        count = 0
        with self:
            for index, (frame, timestamp_ms) in enumerate(frames):
                self.process_frame(frame, index, timestamp_ms)
                count += 1
        return count
