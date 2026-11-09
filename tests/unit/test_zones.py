import numpy as np
import pytest

from movecore.core.config import ConfigError
from movecore.motion.zones import Zone, moving_pixels, parse_zones


def test_zones_are_parsed_from_fractions():
    zones = parse_zones({"left": [0, 0, 0.5, 1], "corner": [0.75, 0.75, 0.25, 0.25]})
    assert zones == (Zone("left", 0, 0, 0.5, 1), Zone("corner", 0.75, 0.75, 0.25, 0.25))


def test_no_zones():
    assert parse_zones(None) == ()
    assert parse_zones({}) == ()


@pytest.mark.parametrize(
    "rect",
    [[0, 0, 0.5], [0, 0, "half", 1], [0, 0, True, 1], [0.6, 0, 0.5, 1], [0, 0, 0, 1], "all"],
)
def test_invalid_zones_are_rejected(rect):
    with pytest.raises(ConfigError, match="Zone 'bad'"):
        parse_zones({"bad": rect})


def test_zones_must_be_a_mapping():
    with pytest.raises(ConfigError, match="must map names"):
        parse_zones([[0, 0, 1, 1]])


def test_to_pixels():
    assert Zone("left", 0, 0, 0.5, 1).to_pixels(640, 480) == (0, 0, 320, 480)
    assert Zone("corner", 0.75, 0.5, 0.25, 0.5).to_pixels(640, 480) == (480, 240, 160, 240)


def test_moving_pixels_counts_only_inside_the_zone():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:, :25] = 255

    assert moving_pixels(mask, (0, 0, 50, 100)) == 2500
    assert moving_pixels(mask, (50, 0, 50, 100)) == 0
