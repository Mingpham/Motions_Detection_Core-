import pytest

from movecore.capture import SourceError, VideoFileSource, selection
from movecore.capture.camera import CameraInfo

INPUT = {"width": 640, "height": 480}
CAMERAS = [CameraInfo("Camera 🥀", "33F1"), CameraInfo("Camera FaceTime HD", "EAB7")]


@pytest.fixture
def opened_cameras(monkeypatch):
    """Record CameraSource calls instead of opening real cameras."""
    calls = []

    def fake_camera(index, width, height, name=None):
        calls.append({"index": index, "size": (width, height), "name": name})
        return "camera"

    monkeypatch.setattr(selection, "CameraSource", fake_camera)
    monkeypatch.setattr(selection, "list_cameras", lambda: CAMERAS)
    return calls


def test_digit_opens_camera_by_index(opened_cameras):
    selection.open_source("2", INPUT)
    assert opened_cameras == [{"index": 2, "size": (640, 480), "name": None}]


def test_name_opens_the_matching_camera(opened_cameras):
    selection.open_source("FaceTime HD", INPUT)
    assert opened_cameras[0]["index"] == 1
    assert opened_cameras[0]["name"] == "Camera FaceTime HD"


def test_config_camera_is_the_default(opened_cameras):
    selection.open_source(None, {**INPUT, "camera": "FaceTime"})
    assert opened_cameras[0]["index"] == 1


def test_camera_zero_when_nothing_is_configured(opened_cameras):
    selection.open_source(None, INPUT)
    assert opened_cameras[0]["index"] == 0


def test_existing_file_opens_as_video(make_video):
    source = selection.open_source(str(make_video()), INPUT, loop=True)
    assert isinstance(source, VideoFileSource)
    assert source.loop
    source.close()


@pytest.mark.parametrize("missing", ["clip.mp4", "data/clips/missing"])
def test_missing_video_path_is_not_treated_as_a_camera_name(opened_cameras, missing):
    with pytest.raises(SourceError, match="Video file not found"):
        selection.open_source(missing, INPUT)
    assert opened_cameras == []


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_source_is_rejected(opened_cameras, bad):
    with pytest.raises(SourceError, match="empty"):
        selection.open_source(bad, INPUT)
    assert opened_cameras == []


def test_superscript_digit_is_not_a_camera_index(opened_cameras):
    with pytest.raises(SourceError, match="No camera matches"):
        selection.open_source("²", INPUT)
