import numpy as np

from movecore.motion.frame_diff import FrameDifferencer


def gray(frame):
    return frame[:, :, 0]


def test_first_frame_has_no_motion(square_frame):
    assert FrameDifferencer().apply(gray(square_frame(100))).max() == 0


def test_still_scene_has_no_motion(square_frame):
    differ = FrameDifferencer()
    differ.apply(gray(square_frame(100)))
    assert differ.apply(gray(square_frame(100))).max() == 0


def test_moving_square_marks_its_leading_and_trailing_edges(square_frame):
    differ = FrameDifferencer()
    differ.apply(gray(square_frame(100)))

    mask = differ.apply(gray(square_frame(120)))

    columns = np.flatnonzero(mask.any(axis=0))
    # The square covered 100-139 and now covers 120-159: 100-119 cleared, 140-159 appeared.
    assert set(columns) == set(range(100, 120)) | set(range(140, 160))


def test_darkening_counts_as_change_too():
    differ = FrameDifferencer(threshold=25)
    differ.apply(np.full((4, 4), 200, np.uint8))
    # With plain uint8 subtraction, 100 - 200 would wrap around instead of being 100.
    assert differ.apply(np.full((4, 4), 100, np.uint8)).min() == 255


def test_changes_below_threshold_are_ignored():
    differ = FrameDifferencer(threshold=25)
    differ.apply(np.full((4, 4), 100, np.uint8))
    assert differ.apply(np.full((4, 4), 120, np.uint8)).max() == 0
