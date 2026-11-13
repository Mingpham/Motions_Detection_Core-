"""The on-screen window."""

from types import TracebackType
from typing import Self

import cv2
import numpy as np

ESC = 27
QUIT_KEYS = {ord("q"), ESC}


class Display:
    """An OpenCV window. On macOS it must be used from the main thread."""

    def __init__(self, title: str = "MoveCore") -> None:
        self.title = title
        self.last_key = -1  # the key pressed during the last show(), or -1 for none
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    def show(self, frame: np.ndarray) -> bool:
        """Show a frame. Return False once the user presses q or Esc, or closes the window."""
        cv2.imshow(self.title, frame)
        # waitKey also runs the window's event loop, so it must be called every frame.
        key = cv2.waitKey(1)
        self.last_key = key & 0xFF if key != -1 else -1
        if self.last_key in QUIT_KEYS:
            return False
        return self._is_open()

    def _is_open(self) -> bool:
        # On macOS a window closed with its red button makes this raise, not return 0.
        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self) -> None:
        cv2.destroyWindow(self.title)
        cv2.waitKey(1)  # lets macOS actually remove the window

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
