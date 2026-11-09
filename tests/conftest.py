import cv2
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def run_in_tmp_dir(tmp_path, monkeypatch):
    """Run every test in its own folder, so relative paths like logs/ never touch the repo."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def square_frame():
    """Build a black BGR frame with a white square at (x, y)."""

    def make(x, y=200, size=40, width=640, height=480):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[y : y + size, x : x + size] = 255
        return frame

    return make


@pytest.fixture
def make_video(tmp_path):
    """Write a tiny clip of a white square moving right, and return its path.

    ``still_frames`` repeats the last position at the end, so motion visibly stops.
    """

    def make(frames=10, width=160, height=120, fps=10.0, name="clip.avi", still_frames=0):
        path = tmp_path / name
        # MJPG in AVI is written by OpenCV itself, so it works on every platform.
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        assert writer.isOpened()
        for i in range(frames + still_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = (min(i, frames - 1) * 10) % (width - 20)
            frame[50:70, x : x + 20] = 255
            writer.write(frame)
        writer.release()
        return path

    return make
