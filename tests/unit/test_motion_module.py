import numpy as np
import pytest

from movecore.core.config import ConfigError
from movecore.core.module import FrameContext
from movecore.motion import MotionModule

# Small values keep the synthetic sequences short; real defaults are in configs/.
FAST = {"method": "diff", "blur": 1, "min_area": 100, "warmup_frames": 0}
FAST |= {"start_frames": 2, "stop_frames": 3}


def run(module, frames):
    """Feed frames 100 ms apart and return (results, events)."""
    module.setup()
    results, events = [], []
    for index, frame in enumerate(frames):
        result = module.process(FrameContext(frame, index, index * 100))
        results.append(result)
        events.extend(result.events)
    return results, events


def still_then_moving_then_still(square_frame, xs):
    return [square_frame(xs[0])] * 3 + [square_frame(x) for x in xs] + [square_frame(xs[-1])] * 5


def test_moving_square_starts_and_stops_motion(square_frame):
    frames = still_then_moving_then_still(square_frame, range(100, 400, 20))

    results, events = run(MotionModule(FAST), frames)

    assert [event.type for event in events] == ["motion.start", "motion.stop"]
    start, stop = events
    assert start.payload["boxes"]
    assert stop.payload["duration_ms"] > 0
    assert results[-1].data["moving"] is False


def test_still_scene_reports_nothing(square_frame):
    results, events = run(MotionModule(FAST), [square_frame(100)] * 10)

    assert events == []
    assert all(result.data["boxes"] == [] for result in results)


def test_no_events_during_warmup(square_frame):
    frames = still_then_moving_then_still(square_frame, range(100, 400, 20))
    _, events = run(MotionModule(FAST | {"warmup_frames": len(frames)}), frames)

    assert events == []


def test_zone_enter_and_leave(square_frame):
    zones = {"left": [0, 0, 0.5, 1], "right": [0.5, 0, 0.5, 1]}
    # Moves only within the left half (x < 320 - 40).
    frames = still_then_moving_then_still(square_frame, range(20, 260, 20))

    _, events = run(MotionModule(FAST | {"zones": zones}), frames)

    zone_events = [(e.type, e.payload["zone"]) for e in events if e.type.startswith("zone.")]
    assert zone_events == [("zone.enter", "left"), ("zone.leave", "left")]


def test_mog2_flags_an_object_that_appears(square_frame):
    empty = np.zeros((480, 640, 3), dtype=np.uint8)
    options = FAST | {"method": "mog2"}

    results, events = run(MotionModule(options), [empty] * 20 + [square_frame(300)] * 3)

    assert [event.type for event in events] == ["motion.start"]
    assert results[-1].data["boxes"]


def test_invalid_options_fail_when_the_module_is_built():
    with pytest.raises(ConfigError):
        MotionModule({"method": "magic"})


def test_draw_marks_boxes_zones_and_label(square_frame):
    module = MotionModule(FAST | {"zones": {"left": [0, 0, 0.5, 1]}})
    results, _ = run(module, [square_frame(x) for x in (100, 120, 140)])
    view = np.zeros((480, 640, 3), dtype=np.uint8)

    module.draw(view, results[-1])

    assert view.any(axis=2).sum() > 0


WALL = (200, 200, 200)


def wall_with_patch(color):
    frame = np.full((480, 640, 3), WALL, dtype=np.uint8)
    frame[150:330, 250:400] = color
    return frame


@pytest.mark.parametrize("method", ["mog2", "knn"])
def test_mid_dark_colored_object_is_not_mistaken_for_a_shadow(method):
    # Orange at about 63% of the wall's brightness: on a gray image this looked like a shadow.
    frames = [wall_with_patch(WALL)] * 20 + [wall_with_patch((40, 110, 190))] * 3
    # KNN marks the whole frame as moving until it has a few samples; warmup hides that.
    results, events = run(MotionModule(FAST | {"method": method, "warmup_frames": 10}), frames)

    assert results[-1].data["boxes"]
    assert [event.type for event in events] == ["motion.start"]


def test_real_shadow_on_the_wall_is_still_ignored():
    # The same gray, only darker: what a shadow does.
    frames = [wall_with_patch(WALL)] * 20 + [wall_with_patch((150, 150, 150))] * 3
    results, events = run(MotionModule(FAST | {"method": "mog2"}), frames)

    assert results[-1].data["boxes"] == []
    assert events == []
