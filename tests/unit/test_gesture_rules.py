import pytest

from movecore.hands.gesture_rules import (
    extended_fingers,
    hand_size,
    is_pinch,
    rule_gesture,
    thumb_direction,
)


def test_open_hand_has_five_fingers_up(hand_points):
    assert extended_fingers(hand_points("1111")) == (True,) * 5
    assert rule_gesture(hand_points("1111"), 0.25) == "fingers_5"


def test_fist_has_none_up(hand_points):
    hand = hand_points("0000", thumb_out=False)
    assert extended_fingers(hand) == (False,) * 5
    assert rule_gesture(hand, 0.25) == "fingers_0"


@pytest.mark.parametrize(
    ("fingers", "thumb_out", "expected"),
    [
        ("1000", False, (False, True, False, False, False)),
        ("1100", False, (False, True, True, False, False)),
        ("0001", True, (True, False, False, False, True)),
    ],
)
def test_each_finger_is_judged_on_its_own(fingers, thumb_out, expected, hand_points):
    assert extended_fingers(hand_points(fingers, thumb_out)) == expected


@pytest.mark.parametrize("turn", [0, 1, 2, 3])
def test_finger_count_does_not_depend_on_rotation(turn, hand_points):
    assert rule_gesture(hand_points("1100", thumb_out=False, turn=turn), 0.25) == "fingers_2"


def test_size_and_position_do_not_matter(hand_points):
    hand = hand_points("1010")
    moved = hand * 3 + (0.5, -0.2, 0.1)
    assert extended_fingers(moved) == extended_fingers(hand)
    assert hand_size(moved) == pytest.approx(3 * hand_size(hand))


def test_thumb_up_and_down(hand_points):
    assert rule_gesture(hand_points("0000", turn=1), 0.25) == "thumb_up"
    assert rule_gesture(hand_points("0000", turn=3), 0.25) == "thumb_down"


def test_thumb_out_sideways_is_just_one_finger(hand_points):
    hand = hand_points("0000")
    assert thumb_direction(hand) is None
    assert rule_gesture(hand, 0.25) == "fingers_1"


def test_pinch_when_thumb_and_index_tips_touch(hand_points):
    hand = hand_points("0111")
    hand[4] = hand[8] + (0.005, 0, 0)

    assert is_pinch(hand, 0.25)
    assert rule_gesture(hand, 0.25) == "pinch"  # pinch wins over the finger count


def test_open_hand_is_not_a_pinch(hand_points):
    assert not is_pinch(hand_points("1111"), 0.25)
