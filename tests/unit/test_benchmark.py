import pytest

from movecore import benchmark
from movecore.benchmark import WARMUP_FRAMES, Measurement, busy_warning, measure, peak_memory_mb
from movecore.core.config import AppConfig


def motion_config():
    return AppConfig(
        input={"width": 160, "height": 120},
        modules={"motion": {"enabled": True, "method": "diff"}},
        output={},
    )


def test_measures_frames_after_the_warm_up(make_video):
    clip = make_video(frames=10)  # shorter than warm-up + frames, so it must loop

    result = measure(motion_config(), str(clip), frames=25)

    assert result.frames == 25
    assert result.seconds > 0
    assert result.fps == pytest.approx(25 / result.seconds)
    assert result.peak_memory_mb > 0
    assert WARMUP_FRAMES > 10


def test_targets():
    assert Measurement(300, 10.0, 400).meets_targets  # 30 FPS, 400 MB
    assert not Measurement(300, 20.0, 400).meets_targets  # 15 FPS
    assert not Measurement(300, 10.0, 2000).meets_targets  # too much memory
    assert Measurement(300, 10.0, 400).ms_per_frame == pytest.approx(1000 / 30)


def test_peak_memory_is_in_megabytes():
    # Any Python process with NumPy and OpenCV loaded uses tens to hundreds of MB.
    assert 10 < peak_memory_mb() < 5000


def test_busy_machine_is_flagged(monkeypatch):
    monkeypatch.setattr(benchmark, "machine_load", lambda: (6.0, 8))
    assert "busy" in busy_warning()

    monkeypatch.setattr(benchmark, "machine_load", lambda: (1.0, 8))
    assert busy_warning() is None
