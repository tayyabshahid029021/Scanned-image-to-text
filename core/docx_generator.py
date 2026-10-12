"""
High-performance OpenXML DOCX generator.
Generates fully compliant Microsoft Word (.docx) documents without third-party binary dependencies.
Conforms strictly to ECMA-376 OpenXML specification.
"""

import io
import zipfile
import html
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any


class DocxGenerator:
    """Generates standard Microsoft Word .docx documents from extracted plain text."""

    @staticmethod
    def _xml_escape(text: str) -> str:
        """Escapes XML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    @classmethod
    def generate_docx_bytes(
        cls,
        text: str,
        title: Optional[str] = "aiTransform Extracted Document",
        subtitle: Optional[str] = "OCR Extraction & Document Transformation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Creates a valid .docx file byte stream from text.
        
        Args:
            text: Plain text content extracted from document.
            title: Document title heading.
            subtitle: Subtitle or metadata line.
            metadata: Additional document metadata dictionary.
        
        Returns:
            bytes: Valid DOCX file byte contents.
        """
        buffer = io.BytesIO()
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as docx:
            # 1. [Content_Types].xml
            content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""
            docx.writestr("[Content_Types].xml", content_types)

            # 2. _rels/.rels
            package_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""
            docx.writestr("_rels/.rels", package_rels)

            # 3. word/_rels/document.xml.rels
            word_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""
            docx.writestr("word/_rels/document.xml.rels", word_rels)

            # 4. word/styles.xml
            styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI" w:cs="Segoe UI"/>
        <w:sz w:val="22"/>
        <w:szCs w:val="22"/>
        <w:color w:val="1F2937"/>
        <w:lang w:val="en-US"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:after="160" w:line="276" w:lineRule="auto"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:pPr>
      <w:spacing w:before="120" w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Segoe UI Semibold" w:hAnsi="Segoe UI Semibold"/>
      <w:b/>
      <w:sz w:val="40"/>
      <w:color w:val="0F172A"/>
    </w:rPr>
  </w:style>
  
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:pPr>
      <w:spacing w:before="0" w:after="240"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI"/>
      <w:i/>
      <w:sz w:val="20"/>
      <w:color w:val="64748B"/>
    </w:rPr>
  </w:style>
</w:styles>"""
            docx.writestr("word/styles.xml", styles_xml)

            # 5. docProps/core.xml
            core_props = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{cls._xml_escape(title or 'aiTransform Extracted Document')}</dc:title>
  <dc:creator>aiTransform OCR Platform (aiTransform.com)</dc:creator>
  <cp:lastModifiedBy>aiTransform.com</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now_iso}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now_iso}</dcterms:modified>
</cp:coreProperties>"""
            docx.writestr("docProps/core.xml", core_props)

            # 6. docProps/app.xml
            app_props = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <Application>aiTransform Scanned Image to Text Extractor</Application>
  <Company>aiTransform.com</Company>
</Properties>"""
            docx.writestr("docProps/app.xml", app_props)

            # 7. word/document.xml
            doc_paragraphs: List[str] = []

            # Add Header / Title
            if title:
                esc_title = cls._xml_escape(title)
                doc_paragraphs.append(f"""<w:p>
  <w:pPr><w:pStyle w:val="Title"/><w:jc w:val="left"/></w:pPr>
  <w:r><w:rPr><w:b/><w:sz w:val="38"/><w:color w:val="0F172A"/></w:rPr><w:t>{esc_title}</w:t></w:r>
</w:p>""")

            # Add Subtitle / Metadata
            if subtitle:
                esc_sub = cls._xml_escape(subtitle)
                doc_paragraphs.append(f"""<w:p>
  <w:pPr><w:pStyle w:val="Subtitle"/></w:pPr>
  <w:r><w:rPr><w:i/><w:sz w:val="20"/><w:color w:val="64748B"/></w:rPr><w:t>{esc_sub}</w:t></w:r>
</w:p>""")

            # Divider line representation
            doc_paragraphs.append("""<w:p>
  <w:pPr>
    <w:pBdr>
      <w:bottom w:val="single" w:sz="6" w:space="4" w:color="CBD5E1"/>
    </w:pBdr>
    <w:spacing w:after="240"/>
  </w:pPr>
</w:p>""")

            # Content paragraphs
            raw_paragraphs = text.split("\n\n")
            for para_block in raw_paragraphs:
                lines = para_block.strip().split("\n")
                para_runs: List[str] = []
                for idx, line in enumerate(lines):
                    clean_line = cls._xml_escape(line.strip())
                    if clean_line:
                        para_runs.append(f'<w:t xml:space="preserve">{clean_line}</w:t>')
                        if idx < len(lines) - 1:
                            para_runs.append("<w:br/>")

                if para_runs:
                    doc_paragraphs.append(f"""<w:p>
  <w:pPr>
    <w:spacing w:after="180" w:line="280" w:lineRule="auto"/>
  </w:pPr>
  <w:r>
    {''.join(para_runs)}
  </w:r>
</w:p>""")

            document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(doc_paragraphs)}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>"""
            docx.writestr("word/document.xml", document_xml)

        buffer.seek(0)
        return buffer.getvalue()


def create_txt_bytes(text: str) -> bytes:
    """Encodes text to standard UTF-8 clean bytes."""
    return text.encode("utf-8")
