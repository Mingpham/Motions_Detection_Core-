from movecore import app
from movecore.core.config import AppConfig
from movecore.core.module import Module, ModuleResult


class Dummy(Module):
    name = "motion"

    def process(self, ctx):
        return ModuleResult()


def test_build_modules_creates_registered_and_reports_missing(monkeypatch):
    monkeypatch.setattr(app, "MODULES", {"motion": Dummy})
    config = AppConfig(
        input={"width": 640, "height": 480},
        modules={"motion": {"enabled": True}, "hands": {"enabled": True}},
        output={},
    )

    modules, missing = app.build_modules(config)

    assert [type(module) for module in modules] == [Dummy]
    assert missing == ["hands"]


def test_average_fps():
    assert app.RunSummary(frames=30, seconds=2.0).average_fps == 15.0
    assert app.RunSummary(frames=0, seconds=0.0).average_fps == 0.0
