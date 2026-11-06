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


def test_prints_config_summary(capsys):
    assert main(["--config", str(DEFAULT_CONFIG), "--source", "clip.mp4"]) == 0

    out = capsys.readouterr().out
    assert "clip.mp4" in out
    assert "640x480" in out
    assert "motion, hands" in out


def test_missing_config_fails_cleanly(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "nope.yaml")]) == 2
    assert "not found" in capsys.readouterr().err
