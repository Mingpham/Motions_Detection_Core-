"""Clean a motion mask and turn it into boxes."""

from typing import NamedTuple

import cv2
import numpy as np


class Box(NamedTuple):
    x: int
    y: int
    width: int
    height: int
    area: int  # of the shape's outline, which is smaller than width * height


def clean_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Remove specks, fill holes, and join nearby parts of the same object."""
    if kernel_size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # erode then dilate: specks vanish
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # dilate then erode: holes close
    return cv2.dilate(mask, kernel, iterations=2)  # a moving arm and body become one shape


def find_boxes(mask: np.ndarray, min_area: int) -> list[Box]:
    """Bounding boxes of the shapes in ``mask`` at least ``min_area`` pixels big, largest first."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, width, height = cv2.boundingRect(contour)
            boxes.append(Box(x, y, width, height, round(area)))
    return sorted(boxes, key=lambda box: box.area, reverse=True)
