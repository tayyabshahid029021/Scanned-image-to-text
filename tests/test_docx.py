"""
Unit tests for core/docx_generator.py.
"""

import io
import zipfile
import xml.etree.ElementTree as ET
import pytest
from core.docx_generator import DocxGenerator, create_txt_bytes


def test_txt_generation():
    text = "Hello from aiTransform OCR."
    txt_bytes = create_txt_bytes(text)
    assert isinstance(txt_bytes, bytes)
    assert txt_bytes.decode("utf-8") == text


def test_docx_zip_structure():
    text = "Line 1 of document.\n\nParagraph 2 of document."
    docx_bytes = DocxGenerator.generate_docx_bytes(
        text=text,
        title="Test Document",
        subtitle="Verification Subtitle",
    )

    assert isinstance(docx_bytes, bytes)
    assert len(docx_bytes) > 500

    # Test that it is a valid zip archive
    zf = zipfile.ZipFile(io.BytesIO(docx_bytes), "r")
    namelist = zf.namelist()

    # Verify standard OpenXML parts
    assert "[Content_Types].xml" in namelist
    assert "_rels/.rels" in namelist
    assert "word/document.xml" in namelist
    assert "word/styles.xml" in namelist
    assert "docProps/core.xml" in namelist
    assert "docProps/app.xml" in namelist


def test_docx_xml_validity():
    text = "Testing special characters: & < > \" ' and multi-line breaks."
    docx_bytes = DocxGenerator.generate_docx_bytes(
        text=text,
        title="XML Safety Test & More",
    )

    zf = zipfile.ZipFile(io.BytesIO(docx_bytes), "r")
    
    # Parse document.xml to ensure XML is well-formed
    doc_xml = zf.read("word/document.xml")
    root = ET.fromstring(doc_xml)
    assert root.tag.endswith("document")

    # Parse core.xml
    core_xml = zf.read("docProps/core.xml")
    core_root = ET.fromstring(core_xml)
    assert core_root.tag.endswith("coreProperties")
