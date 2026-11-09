import pytest

from movecore.core.config import ConfigError
from movecore.motion.settings import MotionSettings


def test_defaults():
    settings = MotionSettings.from_options({"enabled": True})
    assert settings.method == "mog2"
    assert settings.zones == ()


def test_zones_are_parsed():
    settings = MotionSettings.from_options({"zones": {"left": [0, 0, 0.5, 1]}})
    assert settings.zones[0].name == "left"


def test_unknown_option_is_rejected():
    with pytest.raises(ConfigError, match="min_aera"):
        MotionSettings.from_options({"min_aera": 500})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("method", "magic"),
        ("threshold", 300),
        ("threshold", 25.5),
        ("blur", 20),
        ("blur", 0),
        ("kernel_size", 4),
        ("min_area", -1),
        ("start_frames", 0),
        ("history", True),
        ("var_threshold", 0),
        ("detect_shadows", "yes"),
        ("background_ratio", 0),
        ("background_ratio", 1.5),
    ],
)
def test_invalid_values_are_rejected(name, value):
    with pytest.raises(ConfigError, match=f"modules.motion.{name}"):
        MotionSettings.from_options({name: value})
