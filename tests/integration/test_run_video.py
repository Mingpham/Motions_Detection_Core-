"""Run the whole app on generated clips, the way `movecore --source clip.avi` does."""

import json
from pathlib import Path

import yaml

from movecore import app
from movecore.cli import main
from movecore.core.module import Module, ModuleResult

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class FrameRecorder(Module):
    name = "motion"
    seen = []

    def __init__(self, options):
        pass

    def process(self, ctx):
        FrameRecorder.seen.append((ctx.frame.shape, ctx.timestamp_ms))
        return ModuleResult()


def quick_motion_config(tmp_path):
    """The default config with motion tuned to react within a 10-frame clip."""
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["modules"]["motion"] |= {"method": "diff", "warmup_frames": 0, "stop_frames": 3}
    config["modules"]["motion"]["zones"] = {"left": [0, 0, 0.5, 1]}
    path = tmp_path / "quick.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


def test_headless_run_processes_every_frame(make_video, capsys):
    clip = make_video(frames=10)

    assert main(["--config", str(DEFAULT_CONFIG), "--source", str(clip), "--headless"]) == 0

    out = capsys.readouterr().out
    assert "Frames:  160x120 in, 640x480 processed" in out
    assert "Done:    10 frames" in out


def test_loop_with_max_frames(make_video, capsys):
    clip = make_video(frames=4)
    args = ["--config", str(DEFAULT_CONFIG), "--source", str(clip), "--headless", "--loop"]

    assert main([*args, "--max-frames", "9"]) == 0
    assert "Done:    9 frames" in capsys.readouterr().out


def test_modules_receive_preprocessed_frames(make_video, monkeypatch):
    FrameRecorder.seen = []
    monkeypatch.setattr(app, "MODULES", {"motion": FrameRecorder})
    clip = str(make_video(frames=3))

    main(["--config", str(DEFAULT_CONFIG), "--source", clip, "--headless"])

    assert FrameRecorder.seen == [((480, 640, 3), 0), ((480, 640, 3), 100), ((480, 640, 3), 200)]


def test_moving_square_produces_logged_events(make_video, tmp_path, capsys):
    clip = make_video(frames=10, still_frames=6)

    assert (
        main(["--config", quick_motion_config(tmp_path), "--source", str(clip), "--headless"]) == 0
    )

    out = capsys.readouterr().out
    assert "motion.start" in out
    assert "motion.stop" in out
    # The default config logs to logs/events.jsonl, relative to the (temporary) working dir.
    records = [json.loads(line) for line in Path("logs/events.jsonl").read_text().splitlines()]
    types = [record["type"] for record in records]
    assert types[0] == "motion.start"
    assert "zone.enter" in types
    assert types.count("motion.stop") == 1
