import cv2
import numpy as np
import pytest

from movecore.output import Display
from movecore.output.display import ESC

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


@pytest.fixture
def window(monkeypatch):
    """Replace OpenCV's window functions; the returned dict controls what they report."""
    state = {"key": -1, "visible": 1.0, "destroyed": False}

    def get_property(_title, _prop):
        if isinstance(state["visible"], Exception):
            raise state["visible"]
        return state["visible"]

    monkeypatch.setattr(cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: state["key"])
    monkeypatch.setattr(cv2, "getWindowProperty", get_property)
    monkeypatch.setattr(cv2, "destroyWindow", lambda _title: state.update(destroyed=True))
    return state


def test_keeps_running_without_input(window):
    assert Display().show(FRAME)


@pytest.mark.parametrize("key", [ord("q"), ESC])
def test_quit_keys_stop(window, key):
    window["key"] = key
    assert not Display().show(FRAME)


def test_closed_window_stops(window):
    window["visible"] = 0.0
    assert not Display().show(FRAME)


def test_window_that_no_longer_exists_stops(window):
    window["visible"] = cv2.error("window gone")
    assert not Display().show(FRAME)


def test_context_manager_destroys_the_window(window):
    with Display():
        pass
    assert window["destroyed"]
