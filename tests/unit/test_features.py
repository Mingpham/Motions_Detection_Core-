import numpy as np
import pytest

from movecore.hands.features import FEATURE_COUNT, hand_features


def test_63_numbers_with_the_wrist_at_zero(hand_points):
    features = hand_features(hand_points(), "Right")

    assert features.shape == (FEATURE_COUNT,)
    assert features[:3].tolist() == [0, 0, 0]


def test_position_and_size_do_not_matter(hand_points):
    hand = hand_points("1100")
    moved_and_bigger = hand * 2.5 + (0.3, -0.1, 0.05)

    assert hand_features(moved_and_bigger, "Right") == pytest.approx(hand_features(hand, "Right"))


def test_left_hand_looks_like_the_matching_right_hand(hand_points):
    right = hand_points("1010")
    left = right * (-1, 1, 1)  # the same pose with the other hand

    assert hand_features(left, "Left") == pytest.approx(hand_features(right, "Right"))


def test_rotation_is_kept(hand_points):
    # Thumb up and thumb down are the same shape turned over, and must stay different.
    up = hand_features(hand_points("0000", turn=1), "Right")
    down = hand_features(hand_points("0000", turn=3), "Right")

    assert not np.allclose(up, down)
