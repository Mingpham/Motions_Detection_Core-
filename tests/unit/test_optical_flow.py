import cv2
import numpy as np
import pytest

from movecore.motion.optical_flow import (
    dense_flow,
    direction_name,
    flow_to_color,
    screen_angle,
    sparse_flow,
)


@pytest.fixture
def scene():
    """A textured scene and the same scene moved 5 px right and 3 px down."""
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((240, 320)) * 255).astype(np.uint8), (7, 7), 0)
    return base, np.roll(base, (3, 5), axis=(0, 1))


def test_sparse_flow_measures_the_shift(scene):
    points, moves = sparse_flow(*scene, max_corners=100)

    assert len(points) > 50
    assert np.median(moves, axis=0) == pytest.approx((5, 3), abs=0.1)


def test_sparse_flow_only_uses_corners_inside_the_mask(scene):
    mask = np.zeros(scene[0].shape, dtype=np.uint8)
    mask[:, :100] = 255

    points, _ = sparse_flow(*scene, max_corners=100, mask=mask)

    assert len(points) > 0
    assert points[:, 0].max() < 100


def test_sparse_flow_with_nothing_to_track():
    blank = np.zeros((50, 50), dtype=np.uint8)
    points, moves = sparse_flow(blank, blank, max_corners=10)
    assert points.shape == moves.shape == (0, 2)


def test_lost_points_are_dropped(scene):
    # The second frame is unrelated noise: points can't be followed there and back.
    noise = np.random.default_rng(1).integers(0, 255, scene[0].shape, dtype=np.uint8)
    points, _ = sparse_flow(scene[0], noise, max_corners=100)
    assert len(points) < 20


def test_dense_flow_measures_the_shift(scene):
    flow = dense_flow(*scene)

    assert flow.shape == (240, 320, 2)
    assert np.median(flow[40:-40, 40:-40].reshape(-1, 2), axis=0) == pytest.approx((5, 3), abs=0.1)


def test_flow_to_color_is_black_where_nothing_moves():
    velocities = np.zeros((4, 4, 2), dtype=np.float32)
    velocities[0, 0] = (600, 0)

    view = flow_to_color(velocities, full_speed=600)

    assert view[1:, 1:].max() == 0
    assert view[0, 0].max() == 255


@pytest.mark.parametrize(
    ("vx", "vy", "name", "angle"),
    [(10, 0, "right", 0), (0, -10, "up", 90), (-10, 1, "left", 180), (1, 10, "down", 270)],
)
def test_direction_and_angle(vx, vy, name, angle):
    assert direction_name(vx, vy) == name
    assert screen_angle(vx, vy) == pytest.approx(angle, abs=6)
