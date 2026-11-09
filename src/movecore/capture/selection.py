"""Turn a ``--source`` value or config setting into an open frame source."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from movecore.capture.camera import CameraSource, find_camera_index, list_cameras
from movecore.capture.source import FrameSource, SourceError
from movecore.capture.video_file import VideoFileSource

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def open_source(
    source: str | None, input_config: Mapping[str, Any], loop: bool = False
) -> FrameSource:
    """Open a camera index ("0"), a camera name ("FaceTime HD") or a video file path.

    Without ``source``, the config's ``input.camera`` is used, then camera 0.
    """
    width, height = input_config["width"], input_config["height"]
    if source is None:
        source = str(input_config.get("camera", 0))
    if not source.strip():
        raise SourceError("The source is empty; give a camera index, camera name or file.")

    # isdecimal, not isdigit: "²" is a digit to Python but int() rejects it.
    if source.isdecimal():
        return CameraSource(int(source), width, height)

    path = Path(source)
    if path.is_file():
        return VideoFileSource(path, loop=loop)
    if path.suffix.lower() in VIDEO_EXTENSIONS or len(path.parts) > 1:
        raise SourceError(f"Video file not found: {source}")

    cameras = list_cameras()
    index = find_camera_index(source, cameras)
    return CameraSource(index, width, height, name=cameras[index].name)
