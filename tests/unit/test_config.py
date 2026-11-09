from pathlib import Path

import pytest

from movecore.core.config import ConfigError, load_config

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
VALID = """
input: {width: 640, height: 480}
modules:
  motion: {enabled: true}
  hands: {enabled: false}
output: {overlay: true}
"""


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("default.yaml", ["motion", "hands"]),
        ("motion_only.yaml", ["motion"]),
        ("hands_only.yaml", ["hands"]),
    ],
)
def test_shipped_configs_load(filename, expected):
    assert load_config(CONFIGS_DIR / filename).enabled_modules() == expected


def test_enabled_modules_skips_disabled(tmp_path):
    config = load_config(write_config(tmp_path, VALID))
    assert config.enabled_modules() == ["motion"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_malformed_yaml_raises(tmp_path):
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(write_config(tmp_path, "input: [unclosed"))


def test_non_mapping_root_raises(tmp_path):
    with pytest.raises(ConfigError, match="root must be a mapping"):
        load_config(write_config(tmp_path, "- a\n- b\n"))


def test_unsafe_yaml_tag_is_rejected(tmp_path):
    text = "input: !!python/object/apply:os.getcwd []\n"
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(write_config(tmp_path, text))


@pytest.mark.parametrize("width", [0, -640, "640", 640.5, True, None])
def test_invalid_width_raises(tmp_path, width):
    text = VALID.replace("width: 640", f"width: {'null' if width is None else width!r}")
    with pytest.raises(ConfigError, match="input.width"):
        load_config(write_config(tmp_path, text))


def test_module_options_must_be_a_mapping(tmp_path):
    text = VALID.replace("motion: {enabled: true}", "motion: true")
    with pytest.raises(ConfigError, match="modules.motion"):
        load_config(write_config(tmp_path, text))


def test_output_must_be_a_mapping(tmp_path):
    text = VALID.replace("output: {overlay: true}", "output: 1")
    with pytest.raises(ConfigError, match="'output'"):
        load_config(write_config(tmp_path, text))


@pytest.mark.parametrize("camera", ["FaceTime HD", 0, 2])
def test_valid_camera_settings(tmp_path, camera):
    text = VALID.replace("height: 480}", f"height: 480, camera: {camera}}}")
    assert load_config(write_config(tmp_path, text)).input["camera"] == camera


@pytest.mark.parametrize("camera", ['""', -1, "true", "[0]"])
def test_invalid_camera_settings(tmp_path, camera):
    text = VALID.replace("height: 480}", f"height: 480, camera: {camera}}}")
    with pytest.raises(ConfigError, match="input.camera"):
        load_config(write_config(tmp_path, text))


def test_flip_must_be_a_boolean(tmp_path):
    text = VALID.replace("height: 480}", "height: 480, flip: yes please}")
    with pytest.raises(ConfigError, match="input.flip"):
        load_config(write_config(tmp_path, text))
