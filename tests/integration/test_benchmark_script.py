"""The benchmark script end to end: a real child process per config."""

import importlib.util
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark.py"


def load_script():
    spec = importlib.util.spec_from_file_location("benchmark_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_config(path, method="diff"):
    config = {
        "input": {"width": 160, "height": 120},
        "modules": {"motion": {"enabled": True, "method": method}},
        "output": {},
    }
    path.write_text(yaml.safe_dump(config))
    return str(path)


def test_reports_each_config(make_video, tmp_path, capsys):
    clip = make_video(frames=10)
    configs = [write_config(tmp_path / "diff.yaml"), write_config(tmp_path / "mog2.yaml", "mog2")]

    code = load_script().main(["--configs", *configs, "--source", str(clip), "--frames", "20"])

    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if line.startswith(str(tmp_path))]
    assert len(rows) == 2
    assert all(line.endswith(("met", "MISSED")) for line in rows)
    assert code in (0, 1)  # 1 only if this machine is too slow for the targets


def test_errors_in_a_child_are_reported(tmp_path, capsys):
    config = write_config(tmp_path / "c.yaml")

    code = load_script().main(["--configs", config, "--source", "missing.mp4", "--frames", "5"])

    assert code == 1
    assert "Video file not found" in capsys.readouterr().out


def test_missing_config(capsys):
    assert load_script().main(["--configs", "nope.yaml"]) == 2
    assert "config not found" in capsys.readouterr().err
