import numpy as np

from movecore.preprocess import fit_to_size, mirror, preprocess


def test_wide_frame_is_cropped_at_the_sides_not_squashed():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # A 4:3 crop of 1280x720 is 960 wide, so 160 columns go from each side.
    frame[:, :160] = 255
    frame[:, -160:] = 255

    result = fit_to_size(frame, 640, 480)

    assert result.shape == (480, 640, 3)
    assert result.max() == 0


def test_tall_frame_is_cropped_at_top_and_bottom():
    frame = np.zeros((640, 480, 3), dtype=np.uint8)
    # A 4:3 crop of 480x640 is 360 tall, so 140 rows go from top and bottom.
    frame[:140] = 255
    frame[-140:] = 255

    result = fit_to_size(frame, 640, 480)

    assert result.shape == (480, 640, 3)
    assert result.max() == 0


def test_small_frame_is_enlarged():
    assert fit_to_size(np.zeros((240, 320, 3), np.uint8), 640, 480).shape == (480, 640, 3)


def test_frame_of_the_right_size_is_untouched():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert fit_to_size(frame, 640, 480) is frame


def test_mirror_swaps_left_and_right():
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[:, 0] = 255

    result = mirror(frame)

    assert result[:, -1].min() == 255
    assert result[:, 0].max() == 0


def test_preprocess_flips_only_when_asked():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, 0] = 255

    assert preprocess(frame, 640, 480, flip=False)[:, 0].min() == 255
    assert preprocess(frame, 640, 480, flip=True)[:, -1].min() == 255
