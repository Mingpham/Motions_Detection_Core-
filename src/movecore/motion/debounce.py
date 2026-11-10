"""Turn a flickering yes/no per frame into clean start and stop moments."""

from typing import Literal

Change = Literal["start", "stop"] | None


class Debouncer:
    """On after ``start_frames`` "yes" frames in a row, off after ``stop_frames`` "no" frames.

    Without this, one noisy frame would fire a start event and the next a stop event.
    """

    def __init__(self, start_frames: int, stop_frames: int) -> None:
        self.start_frames = start_frames
        self.stop_frames = stop_frames
        self.active = False
        self._streak = 0

    def update(self, detected: bool) -> Change:
        if detected == self.active:
            self._streak = 0
            return None
        self._streak += 1
        needed = self.start_frames if detected else self.stop_frames
        if self._streak < needed:
            return None
        self.active = detected
        self._streak = 0
        return "start" if detected else "stop"


class StableLabel:
    """Accepts a new label only after it has been seen ``frames`` times in a row.

    ``update`` returns True when the accepted label changes; read it from ``current``.
    """

    def __init__(self, frames: int) -> None:
        self.frames = frames
        self.current: str | None = None
        self._candidate: str | None = None
        self._count = 0

    def update(self, label: str | None) -> bool:
        if label == self.current:
            self._candidate, self._count = None, 0
            return False
        if label == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = label, 1
        if self._count < self.frames:
            return False
        self.current, self._candidate, self._count = label, None, 0
        return True
