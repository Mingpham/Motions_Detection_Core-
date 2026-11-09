"""Input sources: cameras and video files."""

from movecore.capture.camera import CameraInfo, CameraSource, list_cameras
from movecore.capture.selection import open_source
from movecore.capture.source import FrameSource, SourceError
from movecore.capture.video_file import VideoFileSource

__all__ = [
    "CameraInfo",
    "CameraSource",
    "FrameSource",
    "SourceError",
    "VideoFileSource",
    "list_cameras",
    "open_source",
]
