import numpy as np
import pytest

from movecore.output import FpsCounter, draw_label


def counter_with_times(*times):
    clock = iter(times)
    return FpsCounter(smoothing=0.9, clock=lambda: next(clock))


def test_first_measurement_is_used_directly():
    counter = counter_with_times(0.0, 0.1)
    counter.tick()
    assert counter.tick() == pytest.approx(10.0)


def test_later_measurements_are_smoothed():
    counter = counter_with_times(0.0, 0.1, 0.15)
    counter.tick()
    counter.tick()
    # 0.9 * 10 FPS + 0.1 * 20 FPS
    assert counter.tick() == pytest.approx(11.0)


def test_clock_that_does_not_move_is_ignored():
    counter = counter_with_times(1.0, 1.0)
    counter.tick()
    assert counter.tick() == 0.0


def test_smoothing_must_be_below_one():
    with pytest.raises(ValueError):
        FpsCounter(smoothing=1.0)


def test_draw_label_marks_the_frame():
    frame = np.full((60, 200, 3), 128, dtype=np.uint8)
    draw_label(frame, "FPS 30.0")

    assert frame.min() == 0
    assert frame.max() == 255
