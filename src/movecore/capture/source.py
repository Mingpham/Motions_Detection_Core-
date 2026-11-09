"""What every input source provides."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from types import TracebackType
from typing import Self

import numpy as np


class SourceError(RuntimeError):
    """A camera or video file could not be opened or read."""


class FrameSource(ABC):
    """A camera or video file that yields frames until it ends or is closed."""

    description: str

    @abstractmethod
    def frames(self) -> Iterator[tuple[np.ndarray, int]]:
        """Yield ``(BGR frame, timestamp_ms)`` with strictly increasing timestamps."""

    @abstractmethod
    def close(self) -> None:
        """Release the device or file."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
