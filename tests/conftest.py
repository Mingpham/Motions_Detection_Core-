import cv2
import numpy as np
import pytest


@pytest.fixture
def make_video(tmp_path):
    """Write a tiny clip of a white square moving right, and return its path."""

    def make(frames=10, width=160, height=120, fps=10.0, name="clip.avi"):
        path = tmp_path / name
        # MJPG in AVI is written by OpenCV itself, so it works on every platform.
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        assert writer.isOpened()
        for i in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = (i * 10) % (width - 20)
            frame[50:70, x : x + 20] = 255
            writer.write(frame)
        writer.release()
        return path

    return make
