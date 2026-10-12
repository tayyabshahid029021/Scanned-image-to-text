"""
Core package for aiTransform Scanned Image to Text Extractor.
"""

from .metrics import calculate_cer, calculate_wer, evaluate_text_accuracy
from .docx_generator import DocxGenerator, create_txt_bytes
from .preprocessor import ImagePreprocessor
from .ocr_engine import OCREngine

__all__ = [
    "calculate_cer",
    "calculate_wer",
    "evaluate_text_accuracy",
    "DocxGenerator",
    "create_txt_bytes",
    "ImagePreprocessor",
    "OCREngine",
]
