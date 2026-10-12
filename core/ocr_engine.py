"""
OCR Engine module powered by RapidOCR ONNX.
Performs text detection, orientation classification, character recognition,
intelligent layout reconstruction, and bounding-box overlay visualization.
"""

import time
import math
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR


class OCREngine:
    """Wrapper around RapidOCR ONNX with document layout analysis and reconstruction."""

    _instance: Optional["OCREngine"] = None

    def __init__(self):
        """Initializes RapidOCR ONNX engine."""
        self.ocr = RapidOCR()

    @classmethod
    def get_instance(cls) -> "OCREngine":
        """Singleton accessor to reuse model sessions across requests."""
        if cls._instance is None:
            cls._instance = OCREngine()
        return cls._instance

    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Runs OCR on image and returns extracted text, layout metadata, and confidence scores.

        Args:
            image: numpy ndarray (BGR or Grayscale)

        Returns:
            Dict containing:
                - full_text: clean reconstructed text with paragraphs and line breaks
                - blocks: list of detected text boxes with coordinates and scores
                - stats: word count, char count, avg confidence, latency
        """
        start_time = time.time()
        
        # Ensure image is in format RapidOCR expects (uint8)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # RapidOCR returns: (results, elapse_list)
        # results is a list of [box_points, text, confidence]
        results, elapse = self.ocr(image)
        total_time = (time.time() - start_time) * 1000.0

        if not results:
            det_t = round(elapse[0] * 1000.0, 2) if (elapse and len(elapse) > 0) else 0.0
            rec_t = round(elapse[2] * 1000.0, 2) if (elapse and len(elapse) > 2) else 0.0
            return {
                "full_text": "",
                "blocks": [],
                "stats": {
                    "word_count": 0,
                    "char_count": 0,
                    "line_count": 0,
                    "block_count": 0,
                    "avg_confidence": 0.0,
                    "min_confidence": 0.0,
                    "latency_ms": round(total_time, 2),
                    "detect_time_ms": det_t,
                    "rec_time_ms": rec_t,
                },
            }

        # Structure blocks
        blocks = []
        confidences = []
        for item in results:
            box = item[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            text = str(item[1]).strip()
            score = float(item[2])
            if not text:
                continue

            confidences.append(score)
            box_arr = np.array(box)
            x_min, y_min = np.min(box_arr, axis=0)
            x_max, y_max = np.max(box_arr, axis=0)
            height = y_max - y_min
            width = x_max - x_min
            center_y = (y_min + y_max) / 2.0
            center_x = (x_min + x_max) / 2.0

            blocks.append({
                "box": box,
                "text": text,
                "confidence": round(score, 4),
                "x_min": float(x_min),
                "y_min": float(y_min),
                "x_max": float(x_max),
                "y_max": float(y_max),
                "center_x": float(center_x),
                "center_y": float(center_y),
                "width": float(width),
                "height": float(height),
            })

        # Reconstruct reading order and layout
        full_text, line_count = self._reconstruct_layout(blocks)

        words = full_text.split()
        avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
        min_conf = min(confidences) if confidences else 0.0

        return {
            "full_text": full_text,
            "blocks": blocks,
            "stats": {
                "word_count": len(words),
                "char_count": len(full_text),
                "line_count": line_count,
                "block_count": len(blocks),
                "avg_confidence": round(avg_conf, 4),
                "min_confidence": round(min_conf, 4),
                "latency_ms": round(total_time, 2),
                "detect_time_ms": round(elapse[0] * 1000.0, 2) if (elapse and len(elapse) > 0) else 0.0,
                "rec_time_ms": round(elapse[2] * 1000.0, 2) if (elapse and len(elapse) > 2) else 0.0,
            },
        }

    def _reconstruct_layout(self, blocks: List[Dict[str, Any]]) -> Tuple[str, int]:
        """
        Sorts bounding boxes into natural reading order:
        Groups items into lines based on vertical overlap,
        orders items left-to-right within lines,
        and detects paragraph spacing.
        """
        if not blocks:
            return "", 0

        # Estimate median line height
        heights = [b["height"] for b in blocks if b["height"] > 0]
        med_height = float(np.median(heights)) if heights else 20.0
        line_overlap_tol = med_height * 0.55

        # Sort blocks initially by center Y
        sorted_blocks = sorted(blocks, key=lambda b: b["center_y"])

        lines: List[List[Dict[str, Any]]] = []
        for block in sorted_blocks:
            placed = False
            for line in lines:
                line_avg_y = sum(b["center_y"] for b in line) / len(line)
                if abs(block["center_y"] - line_avg_y) <= line_overlap_tol:
                    line.append(block)
                    placed = True
                    break
            if not placed:
                lines.append([block])

        # Sort each line horizontally by x_min
        for line in lines:
            line.sort(key=lambda b: b["x_min"])

        # Sort lines by their top-most coordinate
        lines.sort(key=lambda line: min(b["y_min"] for b in line))

        paragraphs: List[str] = []
        current_para_lines: List[str] = []
        prev_line_bottom: Optional[float] = None

        for line in lines:
            line_str = " ".join(b["text"] for b in line)
            line_top = min(b["y_min"] for b in line)
            line_bottom = max(b["y_max"] for b in line)

            if prev_line_bottom is not None:
                vertical_gap = line_top - prev_line_bottom
                # If vertical distance exceeds paragraph threshold, start new paragraph
                if vertical_gap > med_height * 0.9:
                    if current_para_lines:
                        paragraphs.append("\n".join(current_para_lines))
                        current_para_lines = []

            current_para_lines.append(line_str)
            prev_line_bottom = line_bottom

        if current_para_lines:
            paragraphs.append("\n".join(current_para_lines))

        full_document = "\n\n".join(paragraphs)
        return full_document, len(lines)

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        blocks: List[Dict[str, Any]],
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draws bounding boxes and confidence labels over an image for inspection.
        """
        annotated = image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        for b in blocks:
            pts = np.array(b["box"], dtype=np.int32).reshape((-1, 1, 2))
            conf = b["confidence"]

            # Color coding based on confidence:
            # Green for high (>= 0.90), Amber for medium (0.75 - 0.90), Red for low (< 0.75)
            if conf >= 0.90:
                color = (46, 204, 113)  # Green
            elif conf >= 0.75:
                color = (241, 196, 15)  # Amber
            else:
                color = (231, 76, 60)   # Red

            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

            if show_confidence:
                x_min = int(b["x_min"])
                y_min = max(15, int(b["y_min"]))
                label = f"{int(conf * 100)}%"
                cv2.rectangle(annotated, (x_min, y_min - 16), (x_min + 36, y_min), color, -1)
                cv2.putText(
                    annotated,
                    label,
                    (x_min + 2, y_min - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return annotated
