"""Run the whole app on a generated clip, the way `movecore --source clip.avi` does."""

from pathlib import Path

from movecore import app
from movecore.cli import main
from movecore.core.module import Module, ModuleResult

DEFAULT_CONFIG = str(Path(__file__).resolve().parents[2] / "configs" / "default.yaml")


class FrameRecorder(Module):
    name = "motion"
    seen = []

    def process(self, ctx):
        FrameRecorder.seen.append((ctx.frame.shape, ctx.timestamp_ms))
        return ModuleResult()


def test_headless_run_processes_every_frame(make_video, capsys):
    clip = make_video(frames=10)

    assert main(["--config", DEFAULT_CONFIG, "--source", str(clip), "--headless"]) == 0

    out = capsys.readouterr().out
    assert "Frames:  160x120 in, 640x480 processed" in out
    assert "Done:    10 frames" in out


def test_loop_with_max_frames(make_video, capsys):
    clip = make_video(frames=4)
    args = ["--config", DEFAULT_CONFIG, "--source", str(clip), "--headless", "--loop"]

    assert main([*args, "--max-frames", "9"]) == 0
    assert "Done:    9 frames" in capsys.readouterr().out


def test_modules_receive_preprocessed_frames(make_video, monkeypatch):
    FrameRecorder.seen = []
    monkeypatch.setattr(app, "MODULES", {"motion": FrameRecorder})

    main(["--config", DEFAULT_CONFIG, "--source", str(make_video(frames=3)), "--headless"])

    assert FrameRecorder.seen == [((480, 640, 3), 0), ((480, 640, 3), 100), ((480, 640, 3), 200)]
