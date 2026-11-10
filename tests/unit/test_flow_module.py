import numpy as np
import pytest

from movecore.core.config import ConfigError
from movecore.core.module import FrameContext, ModuleResult
from movecore.motion import OpticalFlowModule
from movecore.motion.morphology import Box

FAST = {"stable_frames": 2}


def run(module, frames, motion=None):
    """Feed frames 100 ms apart, optionally with fake motion-module results."""
    module.setup()
    results, events = [], []
    for index, frame in enumerate(frames):
        ctx = FrameContext(frame, index, index * 100)
        if motion is not None:
            ctx.results["motion"] = motion
        result = module.process(ctx)
        results.append(result)
        events.extend(result.events)
    return results, events


@pytest.mark.parametrize("mode", ["sparse", "dense"])
def test_moving_right_is_measured(textured_frame, mode):
    # 8 px every 100 ms = 80 px/s to the right.
    frames = [textured_frame(100 + 8 * i) for i in range(6)]

    results, events = run(OpticalFlowModule(FAST | {"mode": mode}), frames)

    last = results[-1].data
    assert last["direction"] == "right"
    assert last["speed"] == pytest.approx(80, rel=0.15)
    assert last["angle_deg"] == pytest.approx(0, abs=10) or last["angle_deg"] > 350
    assert [(e.type, e.payload["direction"]) for e in events] == [("flow.direction", "right")]


@pytest.mark.parametrize("mode", ["sparse", "dense"])
def test_moving_up_is_measured(textured_frame, mode):
    frames = [textured_frame(300, y=300 - 10 * i) for i in range(6)]
    results, _ = run(OpticalFlowModule(FAST | {"mode": mode}), frames)
    assert results[-1].data["direction"] == "up"


@pytest.mark.parametrize("mode", ["sparse", "dense"])
def test_still_scene_has_no_direction(textured_frame, mode):
    results, events = run(OpticalFlowModule(FAST | {"mode": mode}), [textured_frame(200)] * 5)

    assert results[-1].data["direction"] is None
    assert results[-1].data["speed"] == 0
    assert events == []


def test_slower_than_min_speed_counts_as_still(textured_frame):
    frames = [textured_frame(100 + 2 * i) for i in range(6)]  # 20 px/s
    results, _ = run(OpticalFlowModule(FAST | {"min_speed": 40}), frames)
    assert results[-1].data["direction"] is None


def test_first_frame_is_still(textured_frame):
    results, _ = run(OpticalFlowModule(FAST), [textured_frame(100)])
    assert results[0].data["direction"] is None


@pytest.mark.parametrize("mode", ["sparse", "dense"])
def test_each_motion_box_gets_its_own_velocity(textured_frame, mode):
    frames = [textured_frame(100 + 8 * i) for i in range(4)]
    mask = np.full((480, 640), 255, dtype=np.uint8)
    motion = ModuleResult(data={"mask": mask, "boxes": [Box(80, 180, 160, 120, 19000)]})

    results, _ = run(OpticalFlowModule(FAST | {"mode": mode}), frames, motion)

    box, vx, vy = results[-1].data["boxes"][0]
    assert vx == pytest.approx(80, rel=0.2)
    assert abs(vy) < 10


def test_sparse_ignores_corners_outside_the_motion_mask(textured_frame):
    frames = [textured_frame(100 + 8 * i) for i in range(4)]
    empty_mask = np.zeros((480, 640), dtype=np.uint8)
    motion = ModuleResult(data={"mask": empty_mask, "boxes": []})

    results, _ = run(OpticalFlowModule(FAST), frames, motion)

    assert results[-1].data["direction"] is None


@pytest.mark.parametrize("mode", ["sparse", "dense"])
def test_draw_marks_the_frame(textured_frame, mode):
    module = OpticalFlowModule(FAST | {"mode": mode})
    results, _ = run(module, [textured_frame(100 + 8 * i) for i in range(4)])
    view = np.zeros((480, 640, 3), dtype=np.uint8)

    module.draw(view, results[-1])

    assert view.any()


@pytest.mark.parametrize(
    ("name", "value"),
    [("mode", "magic"), ("max_corners", 0), ("min_speed", 0), ("dense_scale", 1.5), ("speed", 1)],
)
def test_invalid_options_are_rejected(name, value):
    with pytest.raises(ConfigError, match="optical_flow"):
        OpticalFlowModule({name: value})


def test_event_angle_wraps_to_zero(textured_frame):
    # Moving right with a hair of upward drift: 359.x degrees, which must round to 0, not 360.
    frames = [textured_frame(100 + 8 * i, y=200 + (i % 2)) for i in range(6)]
    _, events = run(OpticalFlowModule(FAST), frames)
    assert all(0 <= event.payload["angle_deg"] < 360 for event in events)
