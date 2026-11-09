import numpy as np
import pytest

from movecore.motion.background_sub import BackgroundSubtractor

BACKGROUND = np.full((120, 160), 100, dtype=np.uint8)


def learned(method, **options):
    subtractor = BackgroundSubtractor(method, **options)
    for _ in range(50):
        subtractor.apply(BACKGROUND)
    return subtractor


@pytest.mark.parametrize("method", ["mog2", "knn"])
def test_new_object_is_foreground(method):
    frame = BACKGROUND.copy()
    frame[40:80, 60:100] = 250

    mask = learned(method).apply(frame)

    assert mask[40:80, 60:100].min() == 255
    assert mask[:30].max() == 0


@pytest.mark.parametrize("method", ["mog2", "knn"])
def test_known_background_is_not_foreground(method):
    assert learned(method).apply(BACKGROUND).max() == 0


def test_shadows_are_removed_from_the_mask():
    frame = BACKGROUND.copy()
    frame[40:80, 60:100] = 70  # a bit darker, as a shadow would be

    mask = learned("mog2", detect_shadows=True).apply(frame)

    assert set(np.unique(mask)) <= {0, 255}
    assert mask[40:80, 60:100].max() == 0


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown"):
        BackgroundSubtractor("magic")


def test_lower_background_ratio_keeps_a_still_object_visible_longer():
    # Formula: an object held still joins the background after about
    # ln(ratio) / ln(1 - 1/history) frames: ~5 frames at 0.9, ~34 at 0.5 (history 50).
    def still_after_hold(ratio):
        subtractor = BackgroundSubtractor("mog2", history=50, background_ratio=ratio)
        for _ in range(100):  # past the faster start-up learning
            subtractor.apply(BACKGROUND)
        frame = BACKGROUND.copy()
        frame[40:80, 60:100] = 250
        for _ in range(15):
            mask = subtractor.apply(frame)
        return mask[40:80, 60:100].mean() / 255

    assert still_after_hold(0.9) < 0.1
    assert still_after_hold(0.5) > 0.9
