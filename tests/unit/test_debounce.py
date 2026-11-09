from movecore.motion.debounce import Debouncer


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
