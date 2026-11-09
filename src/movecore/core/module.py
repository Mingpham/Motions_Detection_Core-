"""The contract every detector and analyzer implements."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from movecore.core.events import Event


@dataclass
class ModuleResult:
    """What a module produced for one frame."""

    data: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)


@dataclass
class FrameContext:
    """One frame plus the results of the modules that already ran on it."""

    frame: np.ndarray  # BGR, as delivered by OpenCV
    index: int
    timestamp_ms: int
    results: dict[str, ModuleResult] = field(default_factory=dict)


class Module(ABC):
    """Base class for pipeline modules.

    Subclasses set ``name``, which keys their output in ``FrameContext.results``, and take
    their options from the config (``modules.<name>``) as the only constructor argument.
    """

    name: str

    def setup(self) -> None:  # noqa: B027 - optional hook
        """Load models or allocate state. Called once before the first frame."""

    @abstractmethod
    def process(self, ctx: FrameContext) -> ModuleResult:
        """Analyze one frame."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources. Called once after the last frame."""

    def draw(self, frame: np.ndarray, result: ModuleResult) -> None:  # noqa: B027 - optional hook
        """Draw this module's annotations for one frame onto ``frame``, a display copy."""
