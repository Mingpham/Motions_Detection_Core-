from pathlib import Path

import pytest

from movecore import __version__
from movecore.cli import main

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_missing_source_fails_cleanly(capsys):
    assert main(["--config", str(DEFAULT_CONFIG), "--source", "missing.mp4", "--headless"]) == 1

    captured = capsys.readouterr()
    assert "motion, optical_flow, hands" in captured.out
    assert "Video file not found" in captured.err


def test_max_frames_must_be_positive():
    with pytest.raises(SystemExit) as exit_info:
        main(["--max-frames", "0"])
    assert exit_info.value.code == 2


def test_missing_config_fails_cleanly(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "nope.yaml")]) == 2
    assert "not found" in capsys.readouterr().err
