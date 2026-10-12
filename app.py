"""
aiTransform: Scanned Image to Text Extractor
Web Application powered by Streamlit, OpenCV, and RapidOCR ONNX.
"""

import os
import json
import io
import time
from typing import Dict, Any, Optional
import streamlit as st
import numpy as np
import cv2
from PIL import Image

from core.ocr_engine import OCREngine
from core.preprocessor import ImagePreprocessor
from core.docx_generator import DocxGenerator, create_txt_bytes
from core.metrics import evaluate_text_accuracy

# Configure page layout
st.set_page_config(
    page_title="aiTransform — Scanned Image to Text Extractor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "samples_manifest.json")
BENCHMARK_PATH = os.path.join(BASE_DIR, "data", "benchmark_results.json")

# Custom Styling
st.markdown(
    """
    <style>
    /* aiTransform Theme */
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #2563EB 100%);
        padding: 24px 32px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: #FFFFFF !important;
    }
    .main-header p {
        margin: 6px 0 0 0;
        color: #94A3B8;
        font-size: 1.05rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 6px;
    }
    .badge-blue { background-color: #DBEAFE; color: #1D4ED8; }
    .badge-green { background-color: #DCFCE7; color: #15803D; }
    .badge-purple { background-color: #F3E8FF; color: #7E22CE; }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data
def load_benchmark_data():
    if os.path.exists(BENCHMARK_PATH):
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


manifest = load_manifest()
benchmark_data = load_benchmark_data()


# ------------------------------------------------------------------------------
# Sidebar: Document Input & Preprocessing Controls
# ------------------------------------------------------------------------------
st.sidebar.markdown("### ⚡ aiTransform.com")
st.sidebar.caption("AI Tools & Transformation Products — Task 7 OCR")

# Source selector
source_type = st.sidebar.radio(
    "Choose Input Source",
    ["📁 Upload Image (JPG / PNG)", "🧪 Test Sample Gallery (14 Samples)"],
    index=1 if manifest else 0,
)

current_image: Optional[np.ndarray] = None
image_title = "Uploaded Document"
sample_info = None

if source_type == "🧪 Test Sample Gallery (14 Samples)" and manifest:
    sample_options = [f"[{i['id'][:9]}] {i['quality']} — {i['title'][:32]}..." for i in manifest]
    selected_sample_idx = st.sidebar.selectbox("Select Sample Document", range(len(manifest)), format_func=lambda i: sample_options[i])
    sample_info = manifest[selected_sample_idx]
    image_title = sample_info["title"]
    
    sample_img_path = sample_info["image_path"]
    if os.path.exists(sample_img_path):
        current_image = cv2.imread(sample_img_path)
        # Convert BGR to RGB
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
    
    st.sidebar.info(f"**Degradation:** {sample_info['quality']}\n\n{sample_info['description']}")

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Scanned Document (JPG or PNG)",
        type=["png", "jpg", "jpeg"],
        help="Upload a photo or scan of a printed document.",
    )
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        current_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if current_image is not None:
            current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        image_title = uploaded_file.name

# Preprocessing Toolbar in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Image Cleanup & Preprocessing")

preset = st.sidebar.selectbox(
    "Quick Preset",
    ["Default", "Auto Enhance", "Fix Tilted / Skewed", "Fix Low Light / Dark", "Fix Sideways 90°", "Fix Inverted Dark Mode"],
)

# Preset overrides
preset_rot = 0
preset_deskew = False
preset_clahe = False
preset_shadows = False
preset_invert = False

if preset == "Auto Enhance":
    preset_deskew = True
    preset_clahe = True
    preset_shadows = True
elif preset == "Fix Tilted / Skewed":
    preset_deskew = True
elif preset == "Fix Low Light / Dark":
    preset_clahe = True
    preset_shadows = True
elif preset == "Fix Sideways 90°":
    preset_rot = 270
elif preset == "Fix Inverted Dark Mode":
    preset_invert = True

with st.sidebar.expander("⚙️ Advanced Cleanup Settings", expanded=(preset != "Default")):
    fixed_rot = st.selectbox(
        "Rotate Page",
        [0, 90, 180, 270],
        index=[0, 90, 180, 270].index(preset_rot),
        format_func=lambda x: f"{x}° Clockwise",
    )
    auto_deskew = st.checkbox("Auto-Deskew (Straighten skew)", value=preset_deskew)
    manual_deskew = st.slider("Fine Deskew Angle (°)", -45.0, 45.0, 0.0, step=0.5)
    
    enable_clahe = st.checkbox("CLAHE Contrast Enhancement", value=preset_clahe)
    clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 5.0, 2.5, 0.5) if enable_clahe else 2.5
    
    remove_shadows = st.checkbox("Remove Shadows / Uneven Lighting", value=preset_shadows)
    
    contrast_scale = st.slider("Contrast Multiplier", 0.5, 2.5, 1.0, 0.1)
    brightness_shift = st.slider("Brightness Adjustment", -100, 100, 0, 5)
    
    denoise_level = st.slider("Denoise Strength (Bilateral)", 0, 10, 0, 1)
    sharpen_amount = st.slider("Sharpen Filter", 0.0, 3.0, 0.0, 0.2)
    
    binarize_mode = st.selectbox("Binarization Mode", ["none", "adaptive", "otsu", "grayscale"])
    invert_colors = st.checkbox("Invert Colors (Dark Mode)", value=preset_invert)


# ------------------------------------------------------------------------------
# Main Header
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>aiTransform: Scanned Image to Text Extractor</h1>
        <p>Convert printed document scans and mobile photos into clean, editable text. Download instantly as TXT or Microsoft Word DOCX.</p>
        <div style="margin-top: 10px;">
            <span class="badge-pill badge-blue">✓ OpenCV Computer Vision</span>
            <span class="badge-pill badge-green">✓ RapidOCR ONNX Deep Learning</span>
            <span class="badge-pill badge-purple">✓ ECMA-376 DOCX Export</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_extract, tab_benchmark, tab_guide = st.tabs([
    "📄 Document OCR & Live Preview",
    "📊 Benchmark Evaluation (14 Quality Samples)",
    "📘 Architecture & Technical Notes",
])

# ------------------------------------------------------------------------------
# TAB 1: Document OCR & Live Preview
# ------------------------------------------------------------------------------
with tab_extract:
    if current_image is None:
        st.info("👈 Please upload an image or choose one of the 14 sample documents from the sidebar to begin.")
    else:
        # 1. Run Preprocessing
        bgr_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
        prep_bgr, prep_meta = ImagePreprocessor.process_pipeline(
            bgr_image,
            fixed_rotation=fixed_rot,
            enable_auto_deskew=auto_deskew,
            manual_deskew_angle=manual_deskew,
            enable_clahe=enable_clahe,
            clahe_clip_limit=clahe_clip,
            contrast=contrast_scale,
            brightness=brightness_shift,
            remove_shadows=remove_shadows,
            denoise_level=denoise_level,
            sharpen_amount=sharpen_amount,
            binarization_mode=binarize_mode,
            invert_colors=invert_colors,
        )
        prep_rgb = cv2.cvtColor(prep_bgr, cv2.COLOR_BGR2RGB) if len(prep_bgr.shape) == 3 else prep_bgr

        # 2. Run OCR Engine
        ocr_engine = OCREngine.get_instance()
        with st.spinner("Extracting text via neural OCR engine..."):
            ocr_result = ocr_engine.extract_text(prep_bgr)
        
        extracted_text = ocr_result["full_text"]
        blocks = ocr_result["blocks"]
        stats = ocr_result["stats"]

        # Annotated image with bounding boxes
        annotated_bgr = OCREngine.draw_detections(prep_bgr, blocks)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # Metrics bar
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{stats["word_count"]}</div><div class="metric-lbl">Words</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{stats["char_count"]}</div><div class="metric-lbl">Characters</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{stats["block_count"]}</div><div class="metric-lbl">Detected Blocks</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            conf_pct = int(stats["avg_confidence"] * 100)
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{conf_pct}%</div><div class="metric-lbl">Avg Confidence</div></div>',
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{int(stats["latency_ms"])} ms</div><div class="metric-lbl">Processing Time</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Split screen: Image preview next to extracted text
        col_img, col_txt = st.columns([1, 1], gap="large")

        with col_img:
            st.subheader("🖼️ Document Inspection")
            view_mode = st.radio(
                "Image View",
                ["Original Scan", "Preprocessed (Cleaned)", "OCR Detections Overlay"],
                horizontal=True,
            )

            if view_mode == "Original Scan":
                st.image(current_image, caption=f"Original Input ({image_title})", use_container_width=True)
            elif view_mode == "Preprocessed (Cleaned)":
                st.image(prep_rgb, caption="Cleaned Image after Preprocessing Pipeline", use_container_width=True)
            else:
                st.image(annotated_rgb, caption="Detected Text Blocks with Confidence Scores", use_container_width=True)

            if prep_meta:
                with st.expander("🔍 Applied Preprocessing Metadata"):
                    st.json(prep_meta)

        with col_txt:
            st.subheader("📝 Extracted Text Preview")
            st.caption("You can edit or refine the extracted text before downloading.")
            
            # Editable Text Area
            user_edited_text = st.text_area(
                "Extracted Text",
                value=extracted_text,
                height=480,
                label_visibility="collapsed",
            )

            # Accuracy check against sample ground truth if available
            if sample_info and os.path.exists(sample_info["ground_truth_path"]):
                with open(sample_info["ground_truth_path"], "r", encoding="utf-8") as f:
                    gt_text = f.read().strip()
                eval_acc = evaluate_text_accuracy(gt_text, user_edited_text)
                
                with st.expander("🎯 Accuracy vs Ground Truth (This Sample)"):
                    ca1, ca2, ca3, ca4 = st.columns(4)
                    ca1.metric("Character Accuracy", f"{eval_acc['char_accuracy_pct']}%")
                    ca2.metric("CER", f"{eval_acc['cer']}")
                    ca3.metric("Word Accuracy", f"{eval_acc['word_accuracy_pct']}%")
                    ca4.metric("WER", f"{eval_acc['wer']}")

            st.markdown("#### 📥 Download Extracted Document")
            d_col1, d_col2 = st.columns(2)

            # TXT Download
            txt_bytes = create_txt_bytes(user_edited_text)
            clean_filename = "".join(c for c in image_title if c.isalnum() or c in ("-", "_")).rstrip()
            if not clean_filename:
                clean_filename = "document"

            with d_col1:
                st.download_button(
                    label="📄 Download Clean TXT",
                    data=txt_bytes,
                    file_name=f"{clean_filename}_extracted.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            # DOCX Download
            docx_bytes = DocxGenerator.generate_docx_bytes(
                text=user_edited_text,
                title=image_title.replace("_", " ").title(),
                subtitle="aiTransform.com — Scanned Image to Text Extractor",
            )
            with d_col2:
                st.download_button(
                    label="📘 Download Microsoft DOCX",
                    data=docx_bytes,
                    file_name=f"{clean_filename}_extracted.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )


# ------------------------------------------------------------------------------
# TAB 2: Benchmark Evaluation Suite
# ------------------------------------------------------------------------------
with tab_benchmark:
    st.markdown("### 📊 Benchmark Evaluation Across 14 Quality Variations")
    st.markdown(
        "Evaluation results comparing baseline raw OCR with enhanced preprocessing across "
        "14 distinct physical document degradation profiles."
    )

    if benchmark_data:
        overall = benchmark_data["overall_metrics"]
        samples_res = benchmark_data["samples"]

        # KPI Metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Evaluated Test Set", f"{len(samples_res)} Samples")
        k2.metric("Mean Character Accuracy", f"{overall['average_enhanced_char_accuracy_pct']}%", f"+{overall['overall_cer_improvement']*100:.1f}% vs Raw")
        k3.metric("Average Enhanced CER", f"{overall['average_enhanced_cer']}")
        k4.metric("Mean Latency", f"{overall['average_latency_ms']} ms")

        st.markdown("---")
        st.markdown("#### 📋 Detailed Sample Accuracy Breakdown")

        # Table display
        table_rows = []
        for s in samples_res:
            table_rows.append({
                "Sample": s["title"],
                "Quality / Degradation": s["quality"],
                "Raw CER": f"{s['raw']['cer']:.3f}",
                "Enhanced CER": f"{s['enhanced']['cer']:.3f}",
                "Raw Acc (%)": f"{s['raw']['char_accuracy_pct']}%",
                "Enhanced Acc (%)": f"{s['enhanced']['char_accuracy_pct']}%",
                "Accuracy Gain": f"{s['improvement']['char_acc_delta_pct']:+.1f}%",
                "Latency (ms)": s["enhanced"]["latency_ms"],
            })

        st.dataframe(table_rows, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔍 Qualitative Diagnostic: What Goes Wrong & Why")
        st.caption("Deep-dive failure mode analysis for document degradation patterns.")

        for s in samples_res:
            diag = s["diagnosis"]
            with st.expander(f"📌 {s['quality']} — {s['title']}"):
                col_d1, col_d2 = st.columns([1, 1])
                with col_d1:
                    st.markdown("**⚠️ What Goes Wrong:**")
                    st.write(diag["what_goes_wrong"])
                    st.markdown(f"*Raw CER:* `{s['raw']['cer']}` | *Enhanced CER:* `{s['enhanced']['cer']}`")
                with col_d2:
                    st.markdown("**🛡️ Preprocessing Mitigation:**")
                    st.write(diag["mitigation"])
                    st.markdown(f"*Extracted Characters:* `{s['enhanced']['extracted_chars']}`")

    else:
        st.warning("Benchmark results file not found. Run `python scripts/evaluate.py` to generate.")


# ------------------------------------------------------------------------------
# TAB 3: Architecture & Technical Notes
# ------------------------------------------------------------------------------
with tab_guide:
    st.markdown("### 🏛️ System Architecture & Engineering Approach")
    
    st.markdown(
        """
        #### 1. Computer Vision Preprocessing Pipeline (`core/preprocessor.py`)
        - **Orientation & Deskewing:** Employs Hough Line Transform on Canny edge maps to detect baseline text slopes within $\\pm 45^\\circ$. Applies cubic affine inverse warping with white boundary replication. Fixed 90/180/270 degree orthogonal rotations handle sideways scans.
        - **Contrast & Illumination (CLAHE):** Contrast Limited Adaptive Histogram Equalization operates in the LAB color space L-channel (tile grid $8 \\times 8$, clip limit $2.5$). Avoids global saturation while amplifying faint strokes.
        - **Background Normalization:** Dilates morphological structuring elements across channels to compute illumination backgrounds, normalizing harsh diagonal shadows.
        - **Denoising & Edge Sharpening:** Bilateral edge-preserving filtering eliminates high-frequency salt-and-pepper toner artifacts. Unsharp masking restores softened character glyphs.
        - **Adaptive Binarization:** Local Gaussian window thresholding ($21 \\times 21$ kernel) accommodates paper gradient shifts where global Otsu fails.

        #### 2. Deep Learning OCR Inference (`core/ocr_engine.py`)
        - **Text Detection:** DBNet (Real-time Scene Text Detection with Differentiable Binarization) via ONNX Runtime.
        - **Direction Classification:** Lightweight angle classifier detecting text orientation.
        - **Text Recognition:** SVTR / CRNN deep sequence model recognizing characters.
        - **Layout Reconstruction:** Grouping algorithm clusters detected bounding boxes into lines based on vertical overlap tolerance ($0.55 \\times$ median line height), sorts words horizontally left-to-right, and detects paragraph breaks when vertical gaps exceed $0.9 \\times$ line height.

        #### 3. DOCX OpenXML Document Generation (`core/docx_generator.py`)
        - Pure Python OpenXML (ECMA-376) packaging with zero third-party binary overhead.
        - Generates fully compliant `.docx` packages with `[Content_Types].xml`, `_rels/.rels`, `word/document.xml`, `word/styles.xml`, and core document metadata.
        - Opens seamlessly in Microsoft Word, LibreOffice Writer, Google Docs, and Apple Pages.
        """
    )
