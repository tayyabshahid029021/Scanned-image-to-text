"""
Unit tests for core/preprocessor.py.
"""

import numpy as np
import cv2
import pytest
from core.preprocessor import ImagePreprocessor


def test_rotate_fixed():
    # 100x200 image
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    rot_90 = ImagePreprocessor.rotate_fixed(img, 90)
    assert rot_90.shape == (200, 100, 3)

    rot_180 = ImagePreprocessor.rotate_fixed(img, 180)
    assert rot_180.shape == (100, 200, 3)

    rot_270 = ImagePreprocessor.rotate_fixed(img, 270)
    assert rot_270.shape == (200, 100, 3)


def test_rotate_arbitrary():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    rot = ImagePreprocessor.rotate_arbitrary(img, 45.0)
    assert rot.shape[0] > 100
    assert rot.shape[1] > 100


def test_clahe_contrast():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    enhanced = ImagePreprocessor.apply_clahe(img, clip_limit=3.0)
    assert enhanced.shape == img.shape


def test_denoise():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    denoised = ImagePreprocessor.denoise(img, strength=5)
    assert denoised.shape == img.shape


def test_sharpen():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    sharpened = ImagePreprocessor.sharpen(img, amount=1.5)
    assert sharpened.shape == img.shape


def test_binarize():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    binary = ImagePreprocessor.binarize(img, method="adaptive")
    assert len(binary.shape) == 2
    assert set(np.unique(binary)).issubset({0, 255})


def test_invert():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    inv = ImagePreprocessor.invert(img)
    assert np.all(inv == 255)


def test_process_pipeline():
    img = np.ones((200, 200, 3), dtype=np.uint8) * 200
    processed, meta = ImagePreprocessor.process_pipeline(
        img,
        fixed_rotation=90,
        enable_clahe=True,
        denoise_level=2,
        sharpen_amount=1.0,
    )
    assert processed.shape == (200, 200, 3)
    assert meta["fixed_rotation"] == 90
    assert meta["clahe_applied"] is True
