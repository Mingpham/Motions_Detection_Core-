import numpy as np

from movecore.motion.morphology import Box, clean_mask, find_boxes


def test_specks_are_removed():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10, 10] = mask[50, 70] = 255
    assert clean_mask(mask, kernel_size=5).max() == 0


def test_holes_are_filled():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    mask[49:51, 49:51] = 0
    assert clean_mask(mask, kernel_size=5)[49:51, 49:51].min() == 255


def test_kernel_size_one_leaves_the_mask_alone():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 255
    assert clean_mask(mask, kernel_size=1) is mask


def test_boxes_are_found_largest_first_and_small_ones_dropped():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[10:30, 10:30] = 255  # contour area 19 * 19 = 361
    mask[100:160, 50:110] = 255  # contour area 59 * 59 = 3481
    mask[180:183, 180:183] = 255  # too small

    boxes = find_boxes(mask, min_area=100)

    assert boxes == [Box(50, 100, 60, 60, 3481), Box(10, 10, 20, 20, 361)]


def test_empty_mask_has_no_boxes():
    assert find_boxes(np.zeros((50, 50), dtype=np.uint8), min_area=0) == []
