"""Video file input."""

import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from movecore.capture.source import FrameSource, SourceError

# Used when a file doesn't report a usable frame rate.
DEFAULT_FPS = 30.0


class VideoFileSource(FrameSource):
    def __init__(self, path: str | Path, loop: bool = False) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise SourceError(f"Video file not found: {self.path}")
        self.loop = loop
        self.description = f"video {self.path.name}"
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise SourceError(f"Could not open video file: {self.path}")
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if math.isfinite(fps) and fps > 0 else DEFAULT_FPS

    def frames(self) -> Iterator[tuple[np.ndarray, int]]:
        count = 0
        last_timestamp = -1
        while True:
            read_this_pass = 0
            while True:
                ok, frame = self._capture.read()
                if not ok:
                    break
                read_this_pass += 1
                # Timestamps follow the video's own clock, not how fast we process it,
                # so the same file always gives the same timestamps.
                timestamp_ms = max(round(count * 1000 / self.fps), last_timestamp + 1)
                last_timestamp = timestamp_ms
                count += 1
                yield frame, timestamp_ms

            # A file with no readable frames would otherwise loop forever.
            if not self.loop or read_this_pass == 0:
                return
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self) -> None:
        self._capture.release()
