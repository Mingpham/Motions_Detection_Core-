"""Resize and flip frames into the shape every module expects."""

import cv2
import numpy as np


def fit_to_size(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Center-crop to the target aspect ratio, then resize.

    Resizing 16:9 straight to 4:3 would squash everything sideways and distort hand
    shapes. Cropping keeps proportions and drops the edges instead.
    """
    frame_height, frame_width = frame.shape[:2]
    if (frame_width, frame_height) == (width, height):
        return frame

    target_ratio = width / height
    if frame_width / frame_height > target_ratio:  # too wide: trim left and right
        crop_width = round(frame_height * target_ratio)
        left = (frame_width - crop_width) // 2
        frame = frame[:, left : left + crop_width]
    elif frame_width / frame_height < target_ratio:  # too tall: trim top and bottom
        crop_height = round(frame_width / target_ratio)
        top = (frame_height - crop_height) // 2
        frame = frame[top : top + crop_height]

    # INTER_AREA averages pixels when shrinking, which avoids jagged edges.
    shrinking = frame.shape[1] > width
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def mirror(frame: np.ndarray) -> np.ndarray:
    """Flip left-right, so a webcam image behaves like a mirror."""
    return cv2.flip(frame, 1)


def preprocess(frame: np.ndarray, width: int, height: int, flip: bool) -> np.ndarray:
    frame = fit_to_size(frame, width, height)
    return mirror(frame) if flip else frame
