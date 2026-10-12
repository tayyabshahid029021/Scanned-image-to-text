"""
Computer vision image preprocessing pipeline for scanned document OCR.
Includes deskewing, rotation, contrast enhancement (CLAHE), shadow removal,
noise reduction, sharpening, and adaptive binarization.
"""

import math
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


class ImagePreprocessor:
    """Provides modular image cleanup and enhancement for document OCR."""

    @staticmethod
    def rotate_fixed(image: np.array, angle_degrees: int) -> np.ndarray:
        """
        Rotates image by fixed 90, 180, or 270 degrees.
        """
        angle = angle_degrees % 360
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    @staticmethod
    def rotate_arbitrary(image: np.ndarray, angle: float, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        Rotates image by an arbitrary angle in degrees around center with white background padding.
        """
        if abs(angle) < 0.1:
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # Compute new bounding dimensions
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to take into account translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        border_val = 255 if len(image.shape) == 2 else bg_color
        rotated = cv2.warpAffine(
            image,
            M,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val,
        )
        return rotated

    @classmethod
    def detect_skew_angle(cls, image: np.ndarray) -> float:
        """
        Detects document skew angle using Hough line transform and minimum area rectangle.
        Returns angle in degrees to counter-rotate.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # Resize to max 1200px width for fast robust calculation
        h, w = gray.shape
        scale = 1.0
        if max(h, w) > 1200:
            scale = 1200.0 / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Bilateral filter to reduce noise while keeping edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Probabilistic Hough lines to find text baselines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=15)

        angles = []
        if lines is not None and len(lines) > 0:
            for line in lines:
                pts = line.reshape(-1)
                if len(pts) < 4:
                    continue
                x1, y1, x2, y2 = pts[0], pts[1], pts[2], pts[3]
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                if dx == 0:
                    continue
                angle = math.degrees(math.atan2(dy, dx))
                # Only consider near-horizontal lines (-45 to 45 degrees)
                if -45 < angle < 45:
                    angles.append(angle)

        if angles:
            # Take median angle of dominant lines
            median_angle = float(np.median(angles))
            return median_angle

        # Fallback to minAreaRect on binary text mask
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) > 50:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            else:
                angle = -angle
            if abs(angle) < 45:
                return float(angle)

        return 0.0

    @classmethod
    def auto_deskew(cls, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Automatically detects document skew and straightens the image.
        Returns: (straightened_image, detected_angle)
        """
        angle = cls.detect_skew_angle(image)
        if abs(angle) > 0.3:
            straightened = cls.rotate_arbitrary(image, -angle)
            return straightened, angle
        return image, 0.0

    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.5, tile_grid_size: int = 8) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization.
        Normalizes illumination across shadowed or low-contrast pages.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))

        if len(image.shape) == 3:
            # Convert to LAB color space and apply to L-channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge((l_enhanced, a, b))
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            return clahe.apply(image)

    @staticmethod
    def adjust_contrast_brightness(image: np.ndarray, contrast: float = 1.0, brightness: int = 0) -> np.ndarray:
        """
        Linear contrast and brightness scaling: output = contrast * input + brightness
        contrast: 0.5 to 3.0 (1.0 = neutral)
        brightness: -100 to 100 (0 = neutral)
        """
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    @staticmethod
    def remove_shadows_and_normalize(image: np.ndarray) -> np.ndarray:
        """
        Removes harsh shadows and gradients by estimating background illumination via morphological opening.
        """
        if len(image.shape) == 3:
            # Process each channel
            planes = cv2.split(image)
            result_planes = []
            for plane in planes:
                dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
                bg_img = cv2.medianBlur(dilated_img, 21)
                diff_img = 255 - cv2.absdiff(plane, bg_img)
                norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                result_planes.append(norm_img)
            return cv2.merge(result_planes)
        else:
            dilated_img = cv2.dilate(image, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(image, bg_img)
            return cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    @staticmethod
    def denoise(image: np.ndarray, strength: int = 5) -> np.ndarray:
        """
        Edge-preserving bilateral filter denoising to eliminate scanner grain.
        """
        if strength <= 0:
            return image
        if len(image.shape) == 3:
            return cv2.bilateralFilter(image, d=7, sigmaColor=strength * 10, sigmaSpace=strength * 10)
        else:
            return cv2.bilateralFilter(image, d=7, sigmaColor=strength * 10, sigmaSpace=strength * 10)

    @staticmethod
    def sharpen(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
        """
        Unsharp masking filter to sharpen blurry text contours.
        """
        if amount <= 0.0:
            return image
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return sharpened

    @staticmethod
    def binarize(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
        """
        Converts image to high-contrast binary black & white.
        Methods: 'adaptive', 'otsu', 'grayscale'.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        if method == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
            )
            return binary
        elif method == "grayscale":
            return gray
        return image

    @staticmethod
    def invert(image: np.ndarray) -> np.ndarray:
        """Inverts colors (e.g., for dark mode / white text on dark background)."""
        return cv2.bitwise_not(image)

    @classmethod
    def process_pipeline(
        cls,
        image: np.ndarray,
        fixed_rotation: int = 0,
        enable_auto_deskew: bool = True,
        manual_deskew_angle: float = 0.0,
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        contrast: float = 1.0,
        brightness: int = 0,
        remove_shadows: bool = False,
        denoise_level: int = 0,
        sharpen_amount: float = 0.0,
        binarization_mode: str = "none",  # 'none', 'adaptive', 'otsu', 'grayscale'
        invert_colors: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes a complete configurable image enhancement workflow.
        Returns (processed_image, pipeline_metadata).
        """
        processed = image.copy()
        metadata = {}

        # 1. Color Inversion (if dark mode)
        if invert_colors:
            processed = cls.invert(processed)
            metadata["inverted"] = True

        # 2. Fixed Rotation (0, 90, 180, 270)
        if fixed_rotation % 360 != 0:
            processed = cls.rotate_fixed(processed, fixed_rotation)
            metadata["fixed_rotation"] = fixed_rotation

        # 3. Deskewing
        detected_angle = 0.0
        if enable_auto_deskew:
            processed, detected_angle = cls.auto_deskew(processed)
            metadata["auto_deskew_angle"] = round(detected_angle, 2)
        
        if abs(manual_deskew_angle) > 0.05:
            processed = cls.rotate_arbitrary(processed, manual_deskew_angle)
            metadata["manual_deskew_angle"] = manual_deskew_angle

        # 4. Shadow removal & illumination normalization
        if remove_shadows:
            processed = cls.remove_shadows_and_normalize(processed)
            metadata["shadows_removed"] = True

        # 5. Contrast Enhancement (CLAHE)
        if enable_clahe:
            processed = cls.apply_clahe(processed, clip_limit=clahe_clip_limit)
            metadata["clahe_applied"] = True

        # 6. Linear Contrast & Brightness adjustment
        if contrast != 1.0 or brightness != 0:
            processed = cls.adjust_contrast_brightness(processed, contrast=contrast, brightness=brightness)
            metadata["contrast_adjusted"] = contrast
            metadata["brightness_adjusted"] = brightness

        # 7. Denoising
        if denoise_level > 0:
            processed = cls.denoise(processed, strength=denoise_level)
            metadata["denoised_level"] = denoise_level

        # 8. Sharpening
        if sharpen_amount > 0.0:
            processed = cls.sharpen(processed, amount=sharpen_amount)
            metadata["sharpen_amount"] = sharpen_amount

        # 9. Binarization
        if binarization_mode and binarization_mode != "none":
            processed = cls.binarize(processed, method=binarization_mode)
            metadata["binarization_mode"] = binarization_mode

        return processed, metadata
