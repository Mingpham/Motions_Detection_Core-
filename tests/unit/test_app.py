import numpy as np

from movecore import app
from movecore.core.config import AppConfig
from movecore.core.module import Module, ModuleResult


class Dummy(Module):
    name = "motion"

    def __init__(self, options):
        self.options = options
        self.drawn = 0

    def process(self, ctx):
        return ModuleResult(data={"mask": np.zeros(ctx.frame.shape[:2], np.uint8)})

    def draw(self, frame, result):
        self.drawn += 1


class FakeDisplay:
    """Records what would be shown and quits after three frames."""

    shown = []

    def show(self, frame):
        FakeDisplay.shown.append(frame.shape)
        return len(FakeDisplay.shown) < 3

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        pass


def recording_factory(created):
    """A module factory that keeps every Dummy it builds in ``created``."""

    def build(options):
        created.append(Dummy(options))
        return created[-1]

    return build


def make_config(**output):
    return AppConfig(
        input={"width": 640, "height": 480},
        modules={"motion": {"enabled": True, "min_area": 10}, "hands": {"enabled": True}},
        output={"event_log": None, **output},
    )


def test_build_modules_passes_options_and_reports_missing(monkeypatch):
    monkeypatch.setattr(app, "MODULES", {"motion": Dummy})

    modules, missing = app.build_modules(make_config())

    assert [type(module) for module in modules] == [Dummy]
    assert modules[0].options == {"enabled": True, "min_area": 10}
    assert missing == ["hands"]


def test_motion_modules_are_registered_in_run_order():
    assert list(app.MODULES)[:2] == ["motion", "optical_flow"]


def test_modules_run_in_registry_order_not_config_order(monkeypatch):
    class Flow(Dummy):
        name = "optical_flow"

    monkeypatch.setattr(app, "MODULES", {"motion": Dummy, "optical_flow": Flow})
    config = AppConfig(
        input={"width": 640, "height": 480},
        modules={"optical_flow": {"enabled": True}, "motion": {"enabled": True}},
        output={},
    )

    modules, _ = app.build_modules(config)

    assert [module.name for module in modules] == ["motion", "optical_flow"]


def test_window_shows_drawings_and_the_mask_beside_the_image(monkeypatch, make_video):
    FakeDisplay.shown = []
    modules = []
    monkeypatch.setattr(app, "MODULES", {"motion": recording_factory(modules)})
    monkeypatch.setattr(app, "Display", FakeDisplay)

    summary = app.run(make_config(), str(make_video()), show_mask=True)

    assert summary.frames == 3
    assert FakeDisplay.shown == [(480, 1280, 3)] * 3
    assert modules[0].drawn == 3


def test_overlay_off_skips_module_drawing(monkeypatch, make_video):
    FakeDisplay.shown = []
    modules = []
    monkeypatch.setattr(app, "MODULES", {"motion": recording_factory(modules)})
    monkeypatch.setattr(app, "Display", FakeDisplay)

    app.run(make_config(overlay=False), str(make_video()))

    assert modules[0].drawn == 0
    assert FakeDisplay.shown == [(480, 640, 3)] * 3


def test_average_fps():
    assert app.RunSummary(frames=30, seconds=2.0).average_fps == 15.0
    assert app.RunSummary(frames=0, seconds=0.0).average_fps == 0.0
