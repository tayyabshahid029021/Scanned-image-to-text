"""
Unit tests for core/ocr_engine.py.
"""

import numpy as np
import cv2
import pytest
from core.ocr_engine import OCREngine


def test_ocr_engine_initialization():
    engine = OCREngine.get_instance()
    assert engine is not None
    assert engine.ocr is not None


def test_ocr_blank_image():
    engine = OCREngine.get_instance()
    # Blank white image
    blank = np.ones((200, 400, 3), dtype=np.uint8) * 255
    res = engine.extract_text(blank)
    assert res["full_text"] == ""
    assert res["blocks"] == []
    assert res["stats"]["word_count"] == 0


def test_ocr_with_text():
    engine = OCREngine.get_instance()
    # Draw simple text
    img = np.ones((120, 500, 3), dtype=np.uint8) * 255
    cv2.putText(img, "aiTransform OCR Test", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    res = engine.extract_text(img)

    assert len(res["blocks"]) > 0
    assert res["stats"]["avg_confidence"] > 0.8
    assert "OCR" in res["full_text"] or "aiTransform" in res["full_text"]


def test_draw_detections():
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    blocks = [{
        "box": [[10, 10], [100, 10], [100, 50], [10, 50]],
        "text": "Sample",
        "confidence": 0.95,
        "x_min": 10,
        "y_min": 10,
    }]
    annotated = OCREngine.draw_detections(img, blocks)
    assert annotated.shape == img.shape
