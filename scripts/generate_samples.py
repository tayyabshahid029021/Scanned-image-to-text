"""
Sample Generator Script for aiTransform Scanned Image to Text Extractor.
Generates 14 realistic scanned document images with varied quality
(clear, blurry, tilted, low light, noisy, shadowed, faded, rotated, crumpled, tabular, inverted)
along with exact 1:1 ground truth text files.
"""

import os
import json
import math
from typing import Dict, Any, List
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "data", "ground_truth")

os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)

# Select best available TrueType font
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
]
FONT_PATH = None
for f in FONT_CANDIDATES:
    if os.path.exists(f):
        FONT_PATH = f
        break

BOLD_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    FONT_PATH,
]
BOLD_FONT_PATH = None
for f in BOLD_FONT_CANDIDATES:
    if f and os.path.exists(f):
        BOLD_FONT_PATH = f
        break


def get_font(size: int, bold: bool = False):
    font_file = BOLD_FONT_PATH if bold else FONT_PATH
    if font_file:
        try:
            return ImageFont.truetype(font_file, size)
        except Exception:
            pass
    return ImageFont.load_default()


def render_base_document(
    title: str,
    subtitle: str,
    paragraphs: List[str],
    width: int = 1000,
    height: int = 1300,
    bg_color: tuple = (252, 252, 250),
    text_color: tuple = (25, 30, 40),
    title_size: int = 34,
    body_size: int = 21,
    line_spacing: int = 14,
    para_spacing: int = 26,
    margin_x: int = 70,
    margin_top: int = 75,
) -> Image.Image:
    """Renders a clean, realistic printed document on synthetic paper."""
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    title_font = get_font(title_size, bold=True)
    sub_font = get_font(body_size - 2, bold=False)
    body_font = get_font(body_size, bold=False)

    curr_y = margin_top

    # Header / Title
    draw.text((margin_x, curr_y), title, fill=text_color, font=title_font)
    curr_y += title_size + 14

    # Subtitle
    if subtitle:
        draw.text((margin_x, curr_y), subtitle, fill=(90, 100, 115), font=sub_font)
        curr_y += body_size + 16

    # Header rule line
    draw.line([(margin_x, curr_y), (width - margin_x, curr_y)], fill=(190, 200, 215), width=2)
    curr_y += 24

    max_text_width = width - (margin_x * 2)

    # Paragraphs
    for para in paragraphs:
        words = para.split(" ")
        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            # Measure width
            bbox = draw.textbbox((0, 0), test_line, font=body_font)
            text_w = bbox[2] - bbox[0]
            if text_w <= max_text_width:
                current_line.append(word)
            else:
                if current_line:
                    draw.text((margin_x, curr_y), " ".join(current_line), fill=text_color, font=body_font)
                    curr_y += body_size + line_spacing
                current_line = [word]

        if current_line:
            draw.text((margin_x, curr_y), " ".join(current_line), fill=text_color, font=body_font)
            curr_y += body_size + line_spacing

        curr_y += para_spacing

    # Footer
    footer_text = "aiTransform.com — Enterprise AI Tools & Digital Transformation"
    footer_font = get_font(14, bold=False)
    draw.line([(margin_x, height - 60), (width - margin_x, height - 60)], fill=(220, 225, 230), width=1)
    draw.text((margin_x, height - 48), footer_text, fill=(130, 140, 150), font=footer_font)

    return img


# ==============================================================================
# SAMPLE DEFINITIONS
# ==============================================================================

SAMPLES_SPEC = [
    {
        "id": "sample_01_clear_printed_article",
        "title": "aiTransform Enterprise Briefing: AI Systems 2026",
        "subtitle": "Executive Briefing on Artificial Intelligence and Digital Transformation",
        "quality": "Clear / Baseline",
        "description": "High resolution, high contrast crisp printed article with clear typography.",
        "paragraphs": [
            "Artificial intelligence has moved from experimental laboratories into core enterprise architectures.",
            "Modern organizations deploy autonomous AI agents to automate data ingestion, financial audits, and multi-lingual customer intelligence.",
            "Optical character recognition serves as the foundational bridge connecting legacy paper records with cloud knowledge graphs.",
            "By converting unstructured scans into clean, verifiable structured text, enterprises accelerate digital transformation while preserving historical institutional knowledge.",
        ],
        "degradation": "none",
    },
    {
        "id": "sample_02_slight_blur",
        "title": "aiTransform Research Note: Computer Vision Foundations",
        "subtitle": "Analysis of Deep Learning Architectures in Document Parsing",
        "quality": "Slight Blur",
        "description": "Mild optical defocus blur simulating slight hand tremor or soft focus.",
        "paragraphs": [
            "Computer vision pipelines combine edge detection, convolutional feature maps, and recurrent sequence models.",
            "Modern visual document understanding systems evaluate both spatial geometry and linguistic context simultaneously.",
            "Robust preprocessing transforms degraded sensor inputs into high-fidelity representations before token recognition.",
        ],
        "degradation": "slight_blur",
    },
    {
        "id": "sample_03_heavy_blur",
        "title": "Autonomous Workflow Automation Overview",
        "subtitle": "Operational Resilience and Scalable Infrastructure",
        "quality": "Heavy Blur",
        "description": "Severe motion blur from rapid scanner transit or close-up camera misfocus.",
        "paragraphs": [
            "Real-time processing demands lightweight neural architectures optimized for on-premise execution.",
            "Edge deployment reduces latency while upholding rigorous data privacy guarantees for sensitive enterprise records.",
            "High blur degrades character boundary definition and increases word error rates significantly.",
        ],
        "degradation": "heavy_blur",
    },
    {
        "id": "sample_04_tilted_clockwise_15deg",
        "title": "aiTransform Architecture: Neural OCR Pipeline",
        "subtitle": "Technical Architecture and Preprocessing Flow",
        "quality": "Tilted (+15 deg)",
        "description": "Document tilted clockwise by 15 degrees simulating skewed scanner feed.",
        "paragraphs": [
            "Automated document skew detection employs Radon transforms, Hough line baselines, and minimum bounding rectangles.",
            "Rotational misalignment causes horizontal text bounding boxes to intersect adjacent lines if uncorrected.",
            "Applying affine inverse warping straightens the document plane and restores horizontal reading flow.",
        ],
        "degradation": "tilt_15",
    },
    {
        "id": "sample_05_tilted_counterclockwise_30deg",
        "title": "Technical Specification: Distributed Inference Engine",
        "subtitle": "Cluster Scalability and Latency Benchmarks",
        "quality": "Tilted (-30 deg)",
        "description": "Aggressive -30 degrees counter-clockwise skew from handheld mobile capture.",
        "paragraphs": [
            "Distributed inference clusters divide heavy matrix workloads across heterogeneous tensor cores.",
            "Throughput benchmarks verify sub-second turnaround on full-page document character extraction.",
            "Severe angular displacement requires robust multi-angle detection prior to character segmentation.",
        ],
        "degradation": "tilt_30",
    },
    {
        "id": "sample_06_low_light_underexposed",
        "title": "aiTransform Security Policy: Data Governance Standards",
        "subtitle": "Confidentiality, Integrity, and Access Governance",
        "quality": "Low Light / Underexposed",
        "description": "Captured in dim ambient lighting with underexposed sensor and dark background.",
        "paragraphs": [
            "Enterprise data governance mandates end-to-end encryption for all scanned and digitized artifacts.",
            "Zero trust principles restrict unauthenticated network access to sensitive customer identification documents.",
            "Low light capture compresses image dynamic range, demanding adaptive contrast histogram equalization.",
        ],
        "degradation": "low_light",
    },
    {
        "id": "sample_07_uneven_shadow_gradient",
        "title": "Product Roadmap: Next-Generation Document Processing",
        "subtitle": "Milestones and Strategic Objectives 2026-2027",
        "quality": "Uneven Shadow Gradient",
        "description": "Non-uniform harsh diagonal shadow across the document from mobile phone capture.",
        "paragraphs": [
            "Strategic milestones prioritize intelligent layout decomposition and tabular structure extraction.",
            "Global binarization thresholds fail under non-uniform illumination where shadows darken local regions.",
            "Background illumination normalization flattens lighting gradients before adaptive thresholding.",
        ],
        "degradation": "harsh_shadow",
    },
    {
        "id": "sample_08_noisy_salt_and_pepper",
        "title": "Standard Operating Procedures: Archival Digitization",
        "subtitle": "Quality Assurance Guidelines for Physical Document Scans",
        "quality": "Noisy / Salt & Pepper",
        "description": "High scanner sensor noise, glass dust speckles, and toner artifacting.",
        "paragraphs": [
            "Archival operators must inspect scanner optical plates regularly to prevent dust accumulation.",
            "Impulse noise introduces spurious dark pixels that can be misrecognized as punctuation marks.",
            "Bilateral filtering suppresses high-frequency speckle while retaining sharp text boundary edges.",
        ],
        "degradation": "noisy",
    },
    {
        "id": "sample_09_low_contrast_faded",
        "title": "aiTransform Service Order & Fulfillment Notice",
        "subtitle": "Transaction Record and System Provisioning Confirmation",
        "quality": "Low Contrast / Faded",
        "description": "Faded thermal receipt print or washed out low-toner laser printing.",
        "paragraphs": [
            "Order Reference: TR-8849201. Cloud instance allocation confirmed for enterprise cluster alpha.",
            "Faded printer ribbons produce low optical density characters that blend into paper background.",
            "Dynamic range expansion and CLAHE enhance subtle grayscale gradients to distinguish text glyphs.",
        ],
        "degradation": "faded",
    },
    {
        "id": "sample_10_multi_column_newsletter",
        "title": "aiTransform Insights: The Multi-Column Frontier",
        "subtitle": "Layout Decomposition in Complex Magazine Formats",
        "quality": "Multi-Column Layout",
        "description": "Dual column printed layout testing reading order reconstruction.",
        "paragraphs": [
            "Columnar formatting challenges naive left-to-right text sorters by interweaving separate topical threads.",
            "Modern layout analyzers compute vertical gutters and delineate column boundaries prior to line grouping.",
        ],
        "degradation": "multi_column",
    },
    {
        "id": "sample_11_rotated_90deg",
        "title": "Operational Memo: Landscape Efficiency Metrics",
        "subtitle": "Cross-Departmental Performance Indicators",
        "quality": "Rotated 90 Degrees",
        "description": "Document scanned completely sideways (90 degrees clockwise).",
        "paragraphs": [
            "Key performance indicators highlight a thirty-five percent reduction in manual document entry.",
            "Orientation classifiers determine orthogonal page angles to perform pre-recognition rotation.",
            "Correcting ninety degree orientation is essential for horizontal deep learning text detectors.",
        ],
        "degradation": "rotated_90",
    },
    {
        "id": "sample_12_crumpled_distorted",
        "title": "Field Inspection Checklist: Physical Asset Audit",
        "subtitle": "On-Site Hardware Verification and Compliance",
        "quality": "Crumpled Paper",
        "description": "Creased and crumpled paper texture with warped baselines and shading ripples.",
        "paragraphs": [
            "Physical inspection of rack servers confirmed all biometric sensors remain functional.",
            "Paper creases introduce non-linear geometric warping and linear shading shadows across lines.",
            "Adaptive local filtering mitigates crease boundary artifacts during text segmentation.",
        ],
        "degradation": "crumpled",
    },
    {
        "id": "sample_13_small_font_invoice",
        "title": "aiTransform Solutions: Enterprise Invoice #2026-9041",
        "subtitle": "Billing Summary and Software License Details",
        "quality": "Small Font / Tabular",
        "description": "Fine 9pt condensed font with tabular numeric values and financial codes.",
        "paragraphs": [
            "Invoice Date: October 08, 2026. Account: Enterprise Tier Gold. Currency: USD.",
            "Item 1: aiTransform OCR API License 12 Months - Quantity 1 - Rate 4800.00 - Total 4800.00.",
            "Item 2: Custom Model Fine-Tuning Service - Quantity 40 Hours - Rate 150.00 - Total 6000.00.",
            "Subtotal: 10800.00. Tax: 864.00. Total Balance Due: 11664.00 USD.",
        ],
        "degradation": "small_font",
    },
    {
        "id": "sample_14_inverted_dark_mode",
        "title": "aiTransform Developer Guide: Dark Mode Printout",
        "subtitle": "High-Contrast Terminal Output and Code Documentation",
        "quality": "Inverted / Dark Mode",
        "description": "White and light gray text printed on dark background.",
        "paragraphs": [
            "To initialize the aiTransform Python SDK, import the core OCR client and load your model configuration.",
            "Dark mode scans invert pixel polarities, which breaks standard models trained on black text.",
            "Automatic color inversion normalizes dark mode scans to standard black-on-white polarity.",
        ],
        "degradation": "inverted",
    },
]


def apply_degradation(img_pil: Image.Image, deg_type: str, spec: Dict[str, Any]) -> Image.Image:
    """Applies realistic optical and physical degradations."""
    img_np = np.array(img_pil)

    if deg_type == "slight_blur":
        # Gaussian blur with radius 1.5
        blurred = cv2.GaussianBlur(img_np, (5, 5), 1.6)
        return Image.fromarray(blurred)

    elif deg_type == "heavy_blur":
        # Strong Gaussian blur
        blurred = cv2.GaussianBlur(img_np, (15, 15), 5.0)
        return Image.fromarray(blurred)

    elif deg_type == "tilt_15":
        # Rotate clockwise by 15 deg
        h, w = img_np.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -15.0, 1.0)
        tilted = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return Image.fromarray(tilted)

    elif deg_type == "tilt_30":
        # Rotate counter-clockwise by 30 deg
        h, w = img_np.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 30.0, 0.95)
        tilted = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return Image.fromarray(tilted)

    elif deg_type == "low_light":
        # Dark underexposed capture with noise
        dark = (img_np.astype(np.float32) * 0.28).astype(np.uint8)
        # Add sensor noise
        noise = np.random.normal(0, 12, dark.shape).astype(np.float32)
        noisy_dark = np.clip(dark.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_dark)

    elif deg_type == "harsh_shadow":
        # Diagonal dark shadow gradient
        h, w = img_np.shape[:2]
        X, Y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        # Gradient from top-left bright to bottom-right deep shadow
        gradient = 1.0 - 0.75 * np.clip((X + Y) / 1.5, 0.0, 1.0)
        gradient = np.repeat(gradient[:, :, np.newaxis], 3, axis=2)
        shadowed = (img_np.astype(np.float32) * gradient).astype(np.uint8)
        return Image.fromarray(shadowed)

    elif deg_type == "noisy":
        # Salt & pepper noise + paper grain
        noisy = img_np.copy()
        num_salt = int(0.04 * noisy.size / 3)
        # Salt
        coords = [np.random.randint(0, i - 1, num_salt) for i in noisy.shape[:2]]
        noisy[coords[0], coords[1], :] = 255
        # Pepper
        coords = [np.random.randint(0, i - 1, num_salt) for i in noisy.shape[:2]]
        noisy[coords[0], coords[1], :] = 0
        return Image.fromarray(noisy)

    elif deg_type == "faded":
        # Faded print: low contrast, light text
        gray_tint = (img_np.astype(np.float32) * 0.35 + 165).astype(np.uint8)
        return Image.fromarray(gray_tint)

    elif deg_type == "multi_column":
        # Multi-column layout is drawn specifically
        return img_pil

    elif deg_type == "rotated_90":
        # Rotated 90 deg clockwise
        rot = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
        return Image.fromarray(rot)

    elif deg_type == "crumpled":
        # Simulate paper creases using sine wave modulation and shadows
        h, w = img_np.shape[:2]
        X, Y = np.meshgrid(np.linspace(0, 10 * np.pi, w), np.linspace(0, 10 * np.pi, h))
        ripples = 1.0 + 0.18 * np.sin(X * 0.7 + Y * 0.4) * np.cos(Y * 0.8)
        ripples = np.clip(ripples, 0.75, 1.25)
        crumpled = np.clip(img_np.astype(np.float32) * ripples[:, :, np.newaxis], 0, 255).astype(np.uint8)
        # Add random crease lines
        for _ in range(6):
            pt1 = (np.random.randint(0, w), np.random.randint(0, h))
            pt2 = (np.random.randint(0, w), np.random.randint(0, h))
            cv2.line(crumpled, pt1, pt2, (160, 160, 160), 2)
        return Image.fromarray(crumpled)

    elif deg_type == "small_font":
        return img_pil

    elif deg_type == "inverted":
        # Dark mode: white on dark
        inverted = 255 - img_np
        return Image.fromarray(inverted)

    return img_pil


def generate_all_samples():
    """Generates all sample images and ground truth text files."""
    manifest = []

    for spec in SAMPLES_SPEC:
        sample_id = spec["id"]
        title = spec["title"]
        subtitle = spec["subtitle"]
        paragraphs = spec["paragraphs"]
        deg_type = spec["degradation"]

        # 1. Build Ground Truth Text
        gt_lines = [title]
        if subtitle:
            gt_lines.append(subtitle)
        gt_lines.append("")
        for p in paragraphs:
            gt_lines.append(p)
            gt_lines.append("")
        
        # Footer
        gt_lines.append("aiTransform.com — Enterprise AI Tools & Digital Transformation")
        ground_truth_text = "\n".join(gt_lines).strip()

        # Save Ground Truth File
        gt_file_path = os.path.join(GROUND_TRUTH_DIR, f"{sample_id}.txt")
        with open(gt_file_path, "w", encoding="utf-8") as f:
            f.write(ground_truth_text)

        # 2. Render Document Image
        if deg_type == "small_font":
            base_img = render_base_document(
                title=title,
                subtitle=subtitle,
                paragraphs=paragraphs,
                title_size=28,
                body_size=16,
                line_spacing=10,
                para_spacing=20,
            )
        elif deg_type == "multi_column":
            # Render a 2-column layout
            base_img = Image.new("RGB", (1000, 1300), color=(252, 252, 250))
            draw = ImageDraw.Draw(base_img)
            t_font = get_font(32, bold=True)
            s_font = get_font(18, bold=False)
            b_font = get_font(18, bold=False)

            draw.text((70, 75), title, fill=(25, 30, 40), font=t_font)
            draw.text((70, 120), subtitle, fill=(90, 100, 115), font=s_font)
            draw.line([(70, 155), (930, 155)], fill=(190, 200, 215), width=2)

            # Left column
            col1_x = 70
            col2_x = 520
            col_w = 410

            # Render para 1 in col 1
            y1 = 180
            for line in ["Columnar formatting challenges naive", "left-to-right text sorters by", "interweaving separate topical threads."]:
                draw.text((col1_x, y1), line, fill=(30, 35, 45), font=b_font)
                y1 += 28

            # Render para 2 in col 2
            y2 = 180
            for line in ["Modern layout analyzers compute", "vertical gutters and delineate", "column boundaries prior to line grouping."]:
                draw.text((col2_x, y2), line, fill=(30, 35, 45), font=b_font)
                y2 += 28

            # Divider line between columns
            draw.line([(495, 175), (495, 300)], fill=(220, 225, 230), width=1)

            # Footer
            draw.line([(70, 1240), (930, 1240)], fill=(220, 225, 230), width=1)
            draw.text((70, 1252), "aiTransform.com — Enterprise AI Tools & Digital Transformation", fill=(130, 140, 150), font=get_font(14))

        else:
            base_img = render_base_document(
                title=title,
                subtitle=subtitle,
                paragraphs=paragraphs,
            )

        # Apply degradation
        final_img = apply_degradation(base_img, deg_type, spec)

        # Save Image (PNG)
        img_file_path = os.path.join(SAMPLES_DIR, f"{sample_id}.png")
        final_img.save(img_file_path, "PNG")

        # Also save JPG for format variety requirement (JPG and PNG)
        if "clear" in sample_id or "invoice" in sample_id or "slight_blur" in sample_id:
            jpg_file_path = os.path.join(SAMPLES_DIR, f"{sample_id}.jpg")
            final_img.save(jpg_file_path, "JPEG", quality=95)

        manifest.append({
            "id": sample_id,
            "filename": f"{sample_id}.png",
            "title": title,
            "quality": spec["quality"],
            "description": spec["description"],
            "degradation": deg_type,
            "ground_truth_path": gt_file_path,
            "image_path": img_file_path,
        })

    # Save manifest
    manifest_path = os.path.join(BASE_DIR, "data", "samples_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully generated {len(manifest)} sample images and ground truth files.")
    return manifest


if __name__ == "__main__":
    generate_all_samples()
