"""Webcam input, with cameras chosen by index or by name."""

import json
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from movecore.capture.source import FrameSource, SourceError

SYSTEM_PROFILER = "/usr/sbin/system_profiler"
# macOS cameras often return a few empty frames while they start up.
MAX_MISSED_READS = 50
RETRY_DELAY_SECONDS = 0.05
PERMISSION_HINT = (
    "On macOS, allow camera access for your terminal or VS Code in "
    "System Settings → Privacy & Security → Camera, then restart it."
)


@dataclass(frozen=True)
class CameraInfo:
    name: str
    unique_id: str


def parse_cameras(profiler_json: str) -> list[CameraInfo]:
    """Parse ``system_profiler SPCameraDataType -json`` output into OpenCV's camera order.

    OpenCV's macOS backend numbers cameras by sorting their unique IDs, so sorting the
    same way tells us which index each named camera has.
    """
    entries = json.loads(profiler_json).get("SPCameraDataType", [])
    cameras = [
        CameraInfo(entry.get("_name", ""), entry.get("spcamera_unique-id", "")) for entry in entries
    ]
    return sorted(cameras, key=lambda camera: camera.unique_id)


def list_cameras() -> list[CameraInfo]:
    """Cameras in the order OpenCV numbers them. Only supported on macOS."""
    if sys.platform != "darwin":
        raise SourceError("Listing cameras by name only works on macOS; use an index such as 0.")
    try:
        result = subprocess.run(  # noqa: S603 - fixed command, no user input
            [SYSTEM_PROFILER, "SPCameraDataType", "-json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        return parse_cameras(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise SourceError(f"Could not list cameras: {exc}") from exc


def find_camera_index(name: str, cameras: list[CameraInfo]) -> int:
    """Index of the camera whose name matches exactly, or else contains ``name``."""
    wanted = name.casefold()
    exact = [i for i, camera in enumerate(cameras) if camera.name.casefold() == wanted]
    partial = [i for i, camera in enumerate(cameras) if wanted in camera.name.casefold()]
    matches = exact or partial
    if len(matches) == 1:
        return matches[0]

    found = ", ".join(repr(camera.name) for camera in cameras) or "none"
    if not matches:
        raise SourceError(f"No camera matches {name!r}. Cameras found: {found}")
    raise SourceError(f"{name!r} matches more than one camera ({found}); be more specific.")


class CameraSource(FrameSource):
    def __init__(self, index: int, width: int, height: int, name: str | None = None) -> None:
        self.description = f"camera {index}" + (f" ({name})" if name else "")
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self._capture = cv2.VideoCapture(index, backend)
        if not self._capture.isOpened():
            self._capture.release()
            raise SourceError(f"Could not open {self.description}. {PERMISSION_HINT}")
        # A request, not a guarantee: the camera picks the closest size it supports.
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def frames(self) -> Iterator[tuple[np.ndarray, int]]:
        start = time.monotonic()
        last_timestamp = -1
        missed = 0
        received_any = False
        while True:
            ok, frame = self._capture.read()
            if not ok:
                missed += 1
                if missed > MAX_MISSED_READS:
                    hint = "" if received_any else f" {PERMISSION_HINT}"
                    raise SourceError(f"{self.description} stopped delivering frames.{hint}")
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            missed = 0
            received_any = True
            # MediaPipe's VIDEO mode rejects repeated timestamps, so never reuse one.
            timestamp_ms = max(round((time.monotonic() - start) * 1000), last_timestamp + 1)
            last_timestamp = timestamp_ms
            yield frame, timestamp_ms

    def close(self) -> None:
        self._capture.release()
