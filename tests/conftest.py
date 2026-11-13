from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"

# For hand_points: directions in world landmark space (y points down, -z toward the camera)
# and the four finger knuckles of an upright right hand, in metres from the wrist.
UP, BACK, SIDE = np.array([0, -1, 0]), np.array([0, 0, -1]), np.array([-1, 0, 0])
KNUCKLES = {5: (-0.025, -0.085), 9: (0.0, -0.09), 13: (0.02, -0.085), 17: (0.04, -0.075)}


@pytest.fixture(autouse=True)
def run_in_tmp_dir(tmp_path, monkeypatch):
    """Run every test in its own folder, so relative paths like logs/ never touch the repo."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def default_config(tmp_path):
    """Write configs/default.yaml to a temp file, optionally edited, and return its path.

    Tests run in a temp dir, so the hands module is pointed at the repo's models. They're
    downloaded, not in git, so where they're missing (CI) the hands module is turned off.
    """

    def make(edit=None):
        config = yaml.safe_load((REPO / "configs" / "default.yaml").read_text(encoding="utf-8"))
        hands = config["modules"]["hands"]
        if (MODELS / "gesture_recognizer.task").is_file():
            hands["recognizer_model"] = str(MODELS / "gesture_recognizer.task")
            hands["landmarker_model"] = str(MODELS / "hand_landmarker.task")
        else:
            hands["enabled"] = False
        if edit is not None:
            edit(config)
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return str(path)

    return make


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


@pytest.fixture
def textured_frame():
    """Build a gray BGR frame with a speckled square at (x, y), full of corners to track."""
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur((rng.random((80, 80)) * 255).astype(np.uint8), (5, 5), 0)
    texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

    def make(x, y=200, width=640, height=480):
        frame = np.full((height, width, 3), 90, dtype=np.uint8)
        frame[y : y + 80, x : x + 80] = texture
        return frame

    return make


@pytest.fixture
def hand_points():
    """Build world landmarks (metres, shape (21, 3)) of a hand in a chosen pose."""

    def make(fingers="1111", thumb_out=True, turn=0):
        """World landmarks (metres) of an upright right hand seen from the front.

        ``fingers`` is '0'/'1' for index, middle, ring, pinky. The thumb either sticks out to the
        side or is tucked across the palm. ``turn`` rotates the whole hand by quarter turns,
        clockwise on screen, so a hand with the thumb out and ``turn=1`` is a thumbs-up.
        """
        points = np.zeros((21, 3))
        points[1], points[2] = (-0.03, -0.02, 0), (-0.04, -0.035, 0)
        if thumb_out:
            points[3] = points[2] + 0.03 * SIDE
            points[4] = points[3] + 0.025 * SIDE
        else:
            points[3] = points[2] + (0.007, -0.024, 0)
            points[4] = points[3] + (0.03, 0, 0)  # across the palm, toward the pinky
        for knuckle, up in zip(KNUCKLES, fingers, strict=True):
            points[knuckle] = (*KNUCKLES[knuckle], 0)
            middle = points[knuckle] + (0.04 if up == "1" else 0.035) * UP
            points[knuckle + 1] = middle
            if up == "1":
                points[knuckle + 2] = middle + 0.025 * UP
                points[knuckle + 3] = points[knuckle + 2] + 0.02 * UP
            else:  # bent toward the palm, then back down toward the wrist
                points[knuckle + 2] = middle + 0.02 * BACK
                points[knuckle + 3] = points[knuckle + 2] - 0.025 * UP
        for _ in range(turn):
            x, y = points[:, 0].copy(), points[:, 1].copy()
            points[:, 0], points[:, 1] = -y, x
        return points

    return make
