import json
import subprocess
from itertools import islice

import numpy as np
import pytest

from movecore.capture import SourceError, camera
from movecore.capture.camera import CameraInfo, CameraSource, find_camera_index, parse_cameras

# Shape of `system_profiler SPCameraDataType -json` with an iPhone nearby.
PROFILER_JSON = json.dumps(
    {
        "SPCameraDataType": [
            {"_name": "Camera FaceTime HD", "spcamera_unique-id": "EAB7A68F-EC2B"},
            {"_name": "Camera 🥀", "spcamera_unique-id": "33F190FD-58D0"},
        ]
    }
)
FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


class FakeCapture:
    """Stands in for cv2.VideoCapture: replays a list of read results, then fails."""

    def __init__(self, reads, opened=True):
        self.reads = list(reads)
        self.opened = opened
        self.settings = {}
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.settings[prop] = value

    def read(self):
        if self.reads:
            return (True, FRAME) if self.reads.pop(0) else (False, None)
        return False, None

    def release(self):
        self.released = True


@pytest.fixture
def fake_capture(monkeypatch):
    """Install a FakeCapture and skip real sleeping; returns a setter for its reads."""
    monkeypatch.setattr(camera.time, "sleep", lambda _seconds: None)
    holder = {}

    def install(reads, opened=True):
        capture = FakeCapture(reads, opened)
        monkeypatch.setattr(camera.cv2, "VideoCapture", lambda *_args: capture)
        holder["capture"] = capture
        return capture

    return install


def test_cameras_are_sorted_like_opencv_numbers_them():
    names = [c.name for c in parse_cameras(PROFILER_JSON)]
    assert names == ["Camera 🥀", "Camera FaceTime HD"]


def test_find_camera_by_partial_name():
    cameras = parse_cameras(PROFILER_JSON)
    assert find_camera_index("facetime hd", cameras) == 1


def test_exact_name_wins_over_partial_match():
    cameras = [CameraInfo("Camera", "A"), CameraInfo("Camera FaceTime HD", "B")]
    assert find_camera_index("Camera", cameras) == 0


def test_ambiguous_camera_name_raises():
    with pytest.raises(SourceError, match="more than one"):
        find_camera_index("Camera", parse_cameras(PROFILER_JSON))


def test_unknown_camera_name_lists_what_was_found():
    with pytest.raises(SourceError, match="FaceTime HD"):
        find_camera_index("Logitech", parse_cameras(PROFILER_JSON))


def test_list_cameras_reads_system_profiler(monkeypatch):
    monkeypatch.setattr(camera.sys, "platform", "darwin")
    result = subprocess.CompletedProcess([], 0, stdout=PROFILER_JSON)
    monkeypatch.setattr(camera.subprocess, "run", lambda *_args, **_kwargs: result)

    assert [c.unique_id for c in camera.list_cameras()] == ["33F190FD-58D0", "EAB7A68F-EC2B"]


def test_list_cameras_wraps_profiler_failures(monkeypatch):
    def fail(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "system_profiler")

    monkeypatch.setattr(camera.sys, "platform", "darwin")
    monkeypatch.setattr(camera.subprocess, "run", fail)
    with pytest.raises(SourceError, match="Could not list cameras"):
        camera.list_cameras()


def test_list_cameras_needs_macos(monkeypatch):
    monkeypatch.setattr(camera.sys, "platform", "linux")
    with pytest.raises(SourceError, match="macOS"):
        camera.list_cameras()


def test_camera_that_cannot_open_explains_permission(fake_capture):
    capture = fake_capture([], opened=False)

    with pytest.raises(SourceError, match="Privacy & Security"):
        CameraSource(0, 640, 480)
    assert capture.released


def test_requests_the_configured_size(fake_capture):
    capture = fake_capture([])
    CameraSource(0, 640, 480)

    assert sorted(capture.settings.values()) == [480, 640]


def test_empty_frames_during_startup_are_skipped(fake_capture):
    fake_capture([False, False, False, True, True])

    frames = list(islice(CameraSource(0, 640, 480).frames(), 2))

    assert len(frames) == 2


def test_timestamps_increase_even_when_the_clock_does_not(fake_capture, monkeypatch):
    fake_capture([True, True, True])
    monkeypatch.setattr(camera.time, "monotonic", lambda: 5.0)

    timestamps = [t for _, t in islice(CameraSource(0, 640, 480).frames(), 3)]

    assert timestamps == [0, 1, 2]


def test_no_frames_at_all_raises_with_permission_hint(fake_capture):
    fake_capture([])

    with pytest.raises(SourceError, match="Privacy & Security"):
        list(CameraSource(0, 640, 480).frames())


def test_camera_that_stops_mid_run_raises_without_hint(fake_capture):
    fake_capture([True])
    frames = CameraSource(0, 640, 480).frames()
    next(frames)

    with pytest.raises(SourceError, match="stopped delivering") as error:
        next(frames)
    assert "Privacy" not in str(error.value)
