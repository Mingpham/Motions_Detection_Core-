import numpy as np
import pytest

from movecore.hands.dynamic import (
    PathTracker,
    SwipeDetector,
    hand_size_px,
    index_tip,
    palm_center,
)
from movecore.hands.landmarks import Hand

WIDTH, HEIGHT = 640, 480
STEP_MS = 33  # ~30 FPS


def hand_at(x, y, size=0.1):
    """A hand whose index fingertip is at (x, y), as fractions of the frame, ``size`` tall.

    Hand size in pixels (wrist to middle knuckle) is 0.75 × size × height: 36 px by default.
    """
    image = np.zeros((21, 3))
    image[:, 0], image[:, 1] = x, y + size * 0.6  # palm below the fingertip
    image[0, 1] = y + size * 1.2  # wrist
    image[[5, 9, 13, 17], 1] = y + size * 0.45  # knuckles
    image[8] = (x, y, 0)  # index fingertip
    return Hand("Right", 0.9, image, np.zeros((21, 3)))


def detector(window_ms=250, min_distance=0.8, cooldown_ms=400):
    return SwipeDetector(window_ms, min_distance, cooldown_ms)


def move(swipes, path, start_ms=0, pointing=True):
    """Feed fingertip positions one frame apart; return the swipes reported."""
    results = [
        swipes.update(hand_at(x, y), pointing, start_ms + i * STEP_MS, WIDTH, HEIGHT)
        for i, (x, y) in enumerate(path)
    ]
    return [swipe for swipe in results if swipe is not None]


def line(start, end, frames=6):
    xs, ys = np.linspace(start[0], end[0], frames), np.linspace(start[1], end[1], frames)
    return list(zip(xs, ys, strict=True))


def test_points_and_size_in_pixels():
    hand = hand_at(0.5, 0.5)

    assert index_tip(hand, WIDTH, HEIGHT) == pytest.approx((320, 240))
    assert palm_center(hand, WIDTH, HEIGHT)[1] > 240  # the palm is below the fingertip
    assert hand_size_px(hand, WIDTH, HEIGHT) == pytest.approx(0.075 * HEIGHT)  # 36 px


@pytest.mark.parametrize(
    ("start", "end", "direction"),
    [
        ((0.4, 0.5), (0.5, 0.5), "right"),
        ((0.5, 0.5), (0.4, 0.5), "left"),
        ((0.5, 0.55), (0.5, 0.4), "up"),
        ((0.5, 0.4), (0.5, 0.55), "down"),
    ],
)
def test_a_quick_flick_in_each_direction(start, end, direction):
    # About 64-72 px (~2 hand sizes) in 165 ms.
    (swipe,) = move(detector(), line(start, end))

    assert swipe.direction == direction
    assert swipe.distance >= 0.8
    assert swipe.speed > 3


def test_short_moves_are_not_swipes():
    # 0.03 of the width = 19 px, about half a hand size.
    assert move(detector(), line((0.5, 0.5), (0.53, 0.5))) == []


def test_slow_moves_are_not_swipes():
    # 64 px over 1 s: at most ~16 px fall in the 250 ms window, under 0.8 hand sizes.
    assert move(detector(), line((0.4, 0.5), (0.5, 0.5), frames=30)) == []


def test_only_a_pointing_finger_swipes():
    # A curled index finger moving fast is not a swipe (e.g. a fist being waved).
    assert move(detector(), line((0.4, 0.5), (0.5, 0.5)), pointing=False) == []


def test_opening_a_fist_is_not_an_up_swipe():
    # The fingertip shoots up while the finger straightens: curled for most of the move,
    # extended only at the end. The path restarts when the finger is extended.
    swipes = detector()
    path = line((0.5, 0.6), (0.5, 0.45))
    results = [
        swipes.update(hand_at(x, y), i == len(path) - 1, i * STEP_MS, WIDTH, HEIGHT)
        for i, (x, y) in enumerate(path)
    ]
    assert not any(results)


def test_diagonals_are_not_swipes():
    assert move(detector(), line((0.4, 0.4), (0.5, 0.52))) == []


def test_zigzag_is_not_a_swipe():
    path = [(0.4 + 0.02 * i, 0.4 if i % 2 else 0.5) for i in range(8)]
    assert move(detector(), path) == []


def test_distance_is_in_hand_sizes():
    # The same 64 px flick is ~1.8 hand sizes for a far (small) hand, 0.4 for a near one.
    def swipes_for(size):
        swipes = detector()
        xs = np.linspace(0.4, 0.5, 6)
        return [
            swipes.update(hand_at(x, 0.5, size), True, i * STEP_MS, WIDTH, HEIGHT)
            for i, x in enumerate(xs)
        ]

    assert any(swipes_for(size=0.1))
    assert not any(swipes_for(size=0.5))


def test_a_long_movement_is_one_swipe():
    # 1 s of steady fast movement outlasts the cooldown, but it's still one swipe.
    assert len(move(detector(), line((0.1, 0.5), (0.9, 0.5), frames=30))) == 1


def test_same_direction_again_after_pausing():
    path = line((0.3, 0.5), (0.4, 0.5)) + [(0.4, 0.5)] * 20 + line((0.4, 0.5), (0.5, 0.5))
    assert [s.direction for s in move(detector(), path)] == ["right", "right"]


def test_back_and_forth_gives_both_directions():
    path = line((0.4, 0.5), (0.5, 0.5)) + [(0.5, 0.5)] * 10 + line((0.5, 0.5), (0.4, 0.5))
    assert [s.direction for s in move(detector(), path)] == ["right", "left"]


def test_reset_forgets_the_path():
    swipes = detector()
    move(swipes, line((0.4, 0.5), (0.43, 0.5), frames=3))
    swipes.reset()

    assert swipes.trail == []


def test_path_tracker_keeps_only_the_window():
    path = PathTracker(window_ms=100)
    for i in range(10):
        path.add(i * 50, (i * 10.0, 0.0), size=20.0)

    assert len(path) == 3  # 350, 400, 450 ms
    assert path.span_ms == 100
    assert path.displacement == (20.0, 0.0)
    assert path.distance() == pytest.approx(1.0)
    assert path.straightness() == pytest.approx(1.0)


def test_path_tracker_straightness_of_a_round_trip():
    path = PathTracker(window_ms=1000)
    for i, x in enumerate([0, 10, 20, 10, 0]):
        path.add(i * 10, (float(x), 0.0), size=10.0)

    assert path.straightness() == 0.0
    assert path.distance() == 0.0
