import numpy as np
import pytest

from movecore.core.events import Event, EventBus
from movecore.core.module import FrameContext, Module, ModuleResult
from movecore.core.pipeline import Pipeline

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


class Recorder(Module):
    """Records its lifecycle and what it saw from earlier modules."""

    def __init__(self, name, log=None, fail_on=None):
        self.name = name
        self.log = log if log is not None else []
        self.fail_on = fail_on
        self.seen_results = []

    def _step(self, step):
        self.log.append(f"{self.name}.{step}")
        if step == self.fail_on:
            raise RuntimeError(f"{self.name} failed in {step}")

    def setup(self):
        self._step("setup")

    def process(self, ctx):
        self._step("process")
        self.seen_results.append(list(ctx.results))
        event = Event("test.seen", self.name, ctx.index, ctx.timestamp_ms)
        return ModuleResult(data={"index": ctx.index}, events=[event])

    def close(self):
        self._step("close")


def test_module_requires_process():
    with pytest.raises(TypeError):
        Module()


def test_later_modules_see_earlier_results():
    detector, analyzer = Recorder("detector"), Recorder("analyzer")
    ctx = Pipeline([detector, analyzer]).process_frame(FRAME, 3, 100)

    assert isinstance(ctx, FrameContext)
    assert detector.seen_results == [[]]
    assert analyzer.seen_results == [["detector"]]
    assert ctx.results["analyzer"].data == {"index": 3}


def test_events_are_published_to_typed_and_wildcard_handlers():
    bus = EventBus()
    typed, everything, other = [], [], []
    bus.subscribe("test.seen", typed.append)
    bus.subscribe("*", everything.append)
    bus.subscribe("motion.start", other.append)

    Pipeline([Recorder("a")], bus).process_frame(FRAME, 0, 0)

    assert [event.source for event in typed] == ["a"]
    assert typed == everything
    assert other == []


def test_run_processes_frames_and_manages_lifecycle():
    log = []
    pipeline = Pipeline([Recorder("a", log), Recorder("b", log)])

    count = pipeline.run([(FRAME, 0), (FRAME, 33)])

    assert count == 2
    assert log == [
        "a.setup",
        "b.setup",
        "a.process",
        "b.process",
        "a.process",
        "b.process",
        "b.close",
        "a.close",
    ]


def test_modules_are_closed_when_processing_fails():
    log = []
    pipeline = Pipeline([Recorder("a", log), Recorder("b", log, fail_on="process")])

    with pytest.raises(RuntimeError, match="b failed"):
        pipeline.run([(FRAME, 0)])

    assert log[-2:] == ["b.close", "a.close"]


def test_failed_setup_closes_only_started_modules():
    log = []
    pipeline = Pipeline([Recorder("a", log), Recorder("b", log, fail_on="setup")])

    with pytest.raises(RuntimeError, match="b failed"):
        pipeline.run([])

    assert log == ["a.setup", "b.setup", "a.close"]


def test_one_failed_close_does_not_skip_the_others():
    log = []
    pipeline = Pipeline([Recorder("a", log), Recorder("b", log, fail_on="close")])

    with pytest.raises(ExceptionGroup):
        pipeline.run([])

    assert log[-2:] == ["b.close", "a.close"]


def test_duplicate_module_names_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        Pipeline([Recorder("a"), Recorder("a")])
