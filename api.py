"""
Starlette REST API for aiTransform OCR Engine.
Provides programmatic access to document text extraction, image preprocessing,
DOCX/TXT file conversion, and benchmark metrics.
"""

import os
import json
import base64
from typing import Dict, Any
import cv2
import numpy as np
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, PlainTextResponse
from starlette.routing import Route
from starlette.requests import Request

from core.ocr_engine import OCREngine
from core.preprocessor import ImagePreprocessor
from core.docx_generator import DocxGenerator, create_txt_bytes
from core.metrics import evaluate_text_accuracy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "samples_manifest.json")
BENCHMARK_PATH = os.path.join(BASE_DIR, "data", "benchmark_results.json")


async def health(request: Request) -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "aiTransform Scanned Image to Text Extractor",
        "engine": "RapidOCR ONNX + OpenCV",
        "version": "1.0.0",
    })


async def get_samples(request: Request) -> JSONResponse:
    """Returns metadata for all 14 sample documents."""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return JSONResponse({"samples": manifest, "count": len(manifest)})
    return JSONResponse({"samples": [], "count": 0})


async def get_benchmark(request: Request) -> JSONResponse:
    """Returns benchmark accuracy results and diagnostics."""
    if os.path.exists(BENCHMARK_PATH):
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
        return JSONResponse(benchmark)
    return JSONResponse({"error": "Benchmark data not generated"}, status_code=404)


async def extract_text_endpoint(request: Request) -> JSONResponse:
    """
    Extracts text from uploaded image (multipart/form-data or JSON base64).
    Accepts preprocessing configuration query/form params.
    """
    try:
        content_type = request.headers.get("content-type", "")
        img_bgr = None
        params: Dict[str, Any] = {}

        if "multipart/form-data" in content_type:
            form = await request.form()
            file = form.get("file")
            if not file:
                return JSONResponse({"error": "No file uploaded"}, status_code=400)
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            params = {
                "fixed_rotation": int(form.get("fixed_rotation", 0)),
                "enable_auto_deskew": form.get("enable_auto_deskew", "true").lower() == "true",
                "enable_clahe": form.get("enable_clahe", "false").lower() == "true",
                "remove_shadows": form.get("remove_shadows", "false").lower() == "true",
                "denoise_level": int(form.get("denoise_level", 0)),
                "sharpen_amount": float(form.get("sharpen_amount", 0.0)),
                "binarization_mode": form.get("binarization_mode", "none"),
                "invert_colors": form.get("invert_colors", "false").lower() == "true",
            }

        elif "application/json" in content_type:
            body = await request.json()
            image_b64 = body.get("image_base64")
            if not image_b64:
                return JSONResponse({"error": "Missing image_base64 parameter"}, status_code=400)
            img_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            params = body.get("params", {})

        if img_bgr is None:
            return JSONResponse({"error": "Failed to decode image"}, status_code=400)

        # Apply preprocessing
        processed_img, prep_meta = ImagePreprocessor.process_pipeline(
            img_bgr,
            fixed_rotation=params.get("fixed_rotation", 0),
            enable_auto_deskew=params.get("enable_auto_deskew", False),
            manual_deskew_angle=params.get("manual_deskew_angle", 0.0),
            enable_clahe=params.get("enable_clahe", False),
            clahe_clip_limit=params.get("clahe_clip_limit", 2.0),
            contrast=params.get("contrast", 1.0),
            brightness=params.get("brightness", 0),
            remove_shadows=params.get("remove_shadows", False),
            denoise_level=params.get("denoise_level", 0),
            sharpen_amount=params.get("sharpen_amount", 0.0),
            binarization_mode=params.get("binarization_mode", "none"),
            invert_colors=params.get("invert_colors", False),
        )

        # Run OCR
        engine = OCREngine.get_instance()
        ocr_result = engine.extract_text(processed_img)

        return JSONResponse({
            "success": True,
            "text": ocr_result["full_text"],
            "stats": ocr_result["stats"],
            "preprocessing": prep_meta,
            "blocks": [
                {
                    "text": b["text"],
                    "confidence": b["confidence"],
                    "box": b["box"],
                }
                for b in ocr_result["blocks"]
            ],
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def download_docx_endpoint(request: Request) -> Response:
    """Converts provided text into a downloadable .docx document."""
    try:
        body = await request.json()
        text = body.get("text", "")
        title = body.get("title", "aiTransform Extracted Document")
        docx_bytes = DocxGenerator.generate_docx_bytes(text=text, title=title)
        
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": 'attachment; filename="extracted_document.docx"'},
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def download_txt_endpoint(request: Request) -> Response:
    """Converts provided text into a clean downloadable .txt document."""
    try:
        body = await request.json()
        text = body.get("text", "")
        txt_bytes = create_txt_bytes(text)
        return Response(
            content=txt_bytes,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="extracted_document.txt"'},
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


routes = [
    Route("/health", health, methods=["GET"]),
    Route("/api/samples", get_samples, methods=["GET"]),
    Route("/api/benchmark", get_benchmark, methods=["GET"]),
    Route("/api/extract", extract_text_endpoint, methods=["POST"]),
    Route("/api/download/docx", download_docx_endpoint, methods=["POST"]),
    Route("/api/download/txt", download_txt_endpoint, methods=["POST"]),
]

app = Starlette(debug=False, routes=routes)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
