"""Catch a broken environment early: these are the imports every milestone relies on."""

import platform
import sys

import pytest


def test_opencv_processes_a_frame():
    import cv2
    import numpy as np

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).shape == (8, 8)


def test_mediapipe_tasks_api_is_available():
    from mediapipe.tasks.python import vision

    assert vision.HandLandmarker
    assert vision.GestureRecognizer


def test_sklearn_and_yaml_import():
    import sklearn
    import yaml

    assert sklearn.__version__
    assert yaml.safe_load("a: 1") == {"a": 1}


@pytest.mark.skipif(sys.platform != "darwin", reason="Rosetta only applies to macOS")
def test_python_runs_natively_on_apple_silicon():
    assert platform.machine() == "arm64"
