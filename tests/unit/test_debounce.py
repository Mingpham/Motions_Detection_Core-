from movecore.core.debounce import Debouncer, StableLabel


def run(debouncer, detections):
    return [debouncer.update(detected) for detected in detections]


def test_starts_only_after_enough_frames_in_a_row():
    assert run(Debouncer(3, 2), [True, True, True]) == [None, None, "start"]


def test_a_single_noisy_frame_does_nothing():
    assert run(Debouncer(2, 2), [True, False, True, False]) == [None, None, None, None]


def test_stops_after_enough_still_frames():
    changes = run(Debouncer(1, 3), [True, False, False, False])
    assert changes == ["start", None, None, "stop"]


def test_a_short_pause_does_not_stop():
    debouncer = Debouncer(1, 3)
    changes = run(debouncer, [True, False, False, True, False, False])
    assert changes == ["start", None, None, None, None, None]
    assert debouncer.active


def test_stable_label_needs_repeats_before_changing():
    label = StableLabel(frames=2)
    changes = [label.update(value) for value in ["left", "left", "left", "right", "left"]]
    assert changes == [False, True, False, False, False]
    assert label.current == "left"


def test_stable_label_can_go_back_to_nothing():
    label = StableLabel(frames=1)
    label.update("up")
    assert label.update(None)
    assert label.current is None
