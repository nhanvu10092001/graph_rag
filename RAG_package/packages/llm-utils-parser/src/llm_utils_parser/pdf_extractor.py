"""PDF extractor — text extraction with OCR fallback chain.

Extraction strategy:
1. PyPDFLoader (fast, text-based PDFs)
2. If text is empty/too short → OCR fallback (Docling/PaddleOCR/DeepSeek)
3. Table extraction via pdfplumber (optional)

The OCR provider is selected from config (extraction.ocr.provider).
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from llm_utils_parser.base import BaseExtractor

logger = logging.getLogger(__name__)

# Threshold: if PyPDF extracts fewer chars than this, trigger OCR
MIN_TEXT_THRESHOLD = 50


class PDFExtractor(BaseExtractor):
    """Extract text from PDF files using a hybrid page-by-page strategy:
    1. Extracts digital text from every page.
    2. Detects scanned pages (no fonts or large background images) and runs full-page OCR.
    3. Detects embedded images on digital pages and runs OCR on them.
    4. Optional Markdown table extraction via pdfplumber.

    Config (under extraction.pdf):
        primary_loader: "pymupdf"    # Options: "pymupdf", "pypdf", "unstructured"
        enable_ocr: true             # Enable OCR for scanned pages and embedded images
        extract_tables: false        # Extract tables via pdfplumber
        min_text_threshold: 50       # Min chars threshold
        ocr_dpi: 150                 # Rendering DPI for scanned pages
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        pdf_cfg = self.config.get("pdf", {})
        if not pdf_cfg and "extraction" in self.config:
            pdf_cfg = self.config.get("extraction", {}).get("pdf", {})
        self.primary_loader = pdf_cfg.get("primary_loader", "pymupdf")
        self.enable_ocr = pdf_cfg.get("enable_ocr", False)
        self.extract_tables = pdf_cfg.get("extract_tables", False)
        self.min_text_threshold = pdf_cfg.get("min_text_threshold") or pdf_cfg.get("min_chars", MIN_TEXT_THRESHOLD)
        self.ocr_dpi = int(pdf_cfg.get("ocr_dpi", 150))

    def extract(self, file_path: Path, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Extract text from PDF using hybrid page-by-page extraction with OCR on images."""
        t0 = time.perf_counter()

        metadata: Dict[str, Any] = {
            "extractor": "pdf",
            "file_size": file_path.stat().st_size,
            "ocr_used": False,
            "ocr_images_count": 0,
            "scanned_pages_count": 0,
        }

        # Try hybrid page-by-page extraction via PyMuPDF first
        try:
            text, pages_count, ocr_stats = self._extract_hybrid_pymupdf(file_path)
            metadata["loader"] = "pymupdf_hybrid"
            metadata["pages"] = pages_count
            metadata.update(ocr_stats)
        except Exception as e:
            logger.warning(f"Hybrid PyMuPDF extraction failed: {e}, falling back to standard loader chain")
            text, loader_name, pages_count = self._extract_text_pdf(file_path)
            metadata["loader"] = loader_name
            metadata["pages"] = pages_count

            # Standard fallback if text is empty/minimal
            if self.enable_ocr and len(text.strip()) < self.min_text_threshold:
                ocr_text, ocr_meta = self._ocr_fallback(file_path)
                if ocr_text:
                    text = ocr_text
                    metadata.update(ocr_meta)
                    metadata["ocr_used"] = True

        # Optional table extraction
        if self.extract_tables:
            tables_md, table_count = self._extract_tables(file_path)
            if tables_md:
                text = text + "\n\n" + tables_md
                metadata["tables_found"] = table_count

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        metadata["extraction_time_ms"] = elapsed_ms

        logger.info(
            "pdf_extractor: %d chars, loader=%s, ocr_used=%s, ocr_images=%d in %.1fms",
            len(text), metadata.get("loader", "unknown"),
            metadata.get("ocr_used", False),
            metadata.get("ocr_images_count", 0),
            elapsed_ms,
        )

        return text, metadata

    # ── Hybrid Page-by-Page Extraction ─────────────────────────────

    def _extract_hybrid_pymupdf(self, file_path: Path) -> Tuple[str, int, Dict[str, Any]]:
        """Process PDF page by page: extract native text and OCR scanned pages / embedded images."""
        import pymupdf

        doc = pymupdf.open(str(file_path))
        page_count = len(doc)
        all_pages_text = []

        ocr_chain = None
        if self.enable_ocr:
            try:
                from llm_utils_parser.ocr import OCRFactory
                ocr_chain = OCRFactory.create_chain(self.config)
            except Exception as e:
                logger.warning(f"Failed to create OCR chain: {e}")

        def _run_ocr(img_bytes: bytes, mime_type: str = "image/png") -> str:
            if not ocr_chain:
                return ""
            for ocr_engine in ocr_chain:
                try:
                    res = ocr_engine.extract_image_bytes(img_bytes, mime_type=mime_type)
                    if res and res.strip():
                        return res.strip()
                except Exception as err:
                    logger.debug(f"OCR engine {ocr_engine} failed on image: {err}")
            return ""

        ocr_images_count = 0
        scanned_pages_count = 0
        ocr_used = False

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            page_text = page.get_text().strip()
            fonts = page.get_fonts()
            images = page.get_images(full=True)
            page_area = page.rect.width * page.rect.height

            # Determine if this page is a full scanned image
            is_scanned = False
            if len(fonts) == 0 and (len(images) > 0 or not page_text):
                is_scanned = True
            elif len(images) > 0 and len(page_text) < self.min_text_threshold:
                is_scanned = True
            elif len(images) == 1 and page_area > 0:
                for img_info in images:
                    xref = img_info[0]
                    for bbox in page.get_image_rects(xref):
                        if (bbox.width * bbox.height) / page_area >= 0.80 and len(page_text) < self.min_text_threshold:
                            is_scanned = True
                            break

            if is_scanned and self.enable_ocr and ocr_chain:
                # Render full page to bitmap and OCR
                logger.info(f"Page {page_num} detected as scanned image, running full-page OCR...")
                pix = page.get_pixmap(dpi=self.ocr_dpi)
                img_bytes = pix.tobytes("png")
                ocr_res = _run_ocr(img_bytes, "image/png")
                if ocr_res:
                    all_pages_text.append(f"--- Page {page_num} ---\n{ocr_res}")
                    ocr_used = True
                    scanned_pages_count += 1
                    ocr_images_count += 1
                elif page_text:
                    all_pages_text.append(f"--- Page {page_num} ---\n{page_text}")
            else:
                # Page has digital text, plus check for any embedded images (diagrams, scans, etc.)
                page_components = []
                if page_text:
                    page_components.append(page_text)

                if self.enable_ocr and ocr_chain and len(images) > 0:
                    for img_idx, img_info in enumerate(images):
                        xref = img_info[0]
                        try:
                            base_img = doc.extract_image(xref)
                            if not base_img:
                                continue
                            img_bytes = base_img["image"]
                            img_ext = base_img["ext"]
                            width = base_img.get("width", 0)
                            height = base_img.get("height", 0)

                            # Filter out tiny icon / divider artifacts (< 64x64)
                            if width < 64 or height < 64:
                                continue

                            mime_type = f"image/{img_ext}"
                            img_ocr = _run_ocr(img_bytes, mime_type)
                            if img_ocr:
                                page_components.append(f"\n[Embedded Image {img_idx + 1} on Page {page_num} OCR]:\n{img_ocr}")
                                ocr_used = True
                                ocr_images_count += 1
                        except Exception as e:
                            logger.debug(f"Failed to extract image xref {xref} on page {page_num}: {e}")

                if page_components:
                    all_pages_text.append(f"--- Page {page_num} ---\n" + "\n\n".join(page_components))

        doc.close()

        full_text = "\n\n".join(all_pages_text)
        stats = {
            "ocr_used": ocr_used,
            "ocr_images_count": ocr_images_count,
            "scanned_pages_count": scanned_pages_count,
        }
        return full_text, page_count, stats

    # ── Text extraction ────────────────────────────────────────────

    def _extract_text_pdf(self, file_path: Path) -> Tuple[str, str, int]:
        """Extract text from PDF using configured loader.

        Returns:
            Tuple of (text, loader_name, page_count).
        """
        if self.primary_loader == "unstructured":
            return self._try_unstructured(file_path)

        # Chain: pymupdf → pypdf → unstructured → pypdf2
        for loader_fn in [self._try_pymupdf, self._try_pypdf, self._try_unstructured, self._try_pypdf2]:
            try:
                text, name, pages = loader_fn(file_path)
                if text.strip():
                    return text, name, pages
                logger.debug("%s returned empty text, trying next", name)
            except Exception as e:
                logger.debug("%s failed: %s, trying next", loader_fn.__name__, e)
        return "", "none", 0

    @staticmethod
    def _try_pymupdf(file_path: Path) -> Tuple[str, str, int]:
        import pymupdf
        doc = pymupdf.open(str(file_path))
        pages = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text)
        page_count = len(doc)
        doc.close()
        return "\n\n".join(pages), "pymupdf", page_count

    @staticmethod
    def _try_pypdf(file_path: Path) -> Tuple[str, str, int]:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        text = "\n\n".join(d.page_content for d in docs)
        return text, "pypdf", len(docs)

    @staticmethod
    def _try_unstructured(file_path: Path) -> Tuple[str, str, int]:
        from langchain_community.document_loaders import UnstructuredPDFLoader
        loader = UnstructuredPDFLoader(str(file_path))
        docs = loader.load()
        text = "\n\n".join(d.page_content for d in docs)
        return text, "unstructured", len(docs)

    @staticmethod
    def _try_pypdf2(file_path: Path) -> Tuple[str, str, int]:
        try:
            import pypdf
        except ImportError:
            try:
                import PyPDF2 as pypdf
            except ImportError:
                raise ImportError("pypdf or PyPDF2 required for PDF extraction")

        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            pages = []
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages.append(page_text)
                except Exception:
                    pages.append(f"[Error extracting page {i + 1}]")

            return "\n\n".join(pages), "pypdf2", len(reader.pages)

    # ── OCR fallback ───────────────────────────────────────────────

    def _ocr_fallback(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Run OCR on the PDF using configured provider chain."""
        try:
            from llm_utils_parser.ocr import OCRFactory

            chain = OCRFactory.create_chain(self.config)
            if not chain:
                logger.warning("OCR is enabled but no OCR provider configured or available")
                return "", {}

            for ocr in chain:
                try:
                    text, meta = ocr.extract_text(file_path)
                    if len(text.strip()) >= self.min_text_threshold:
                        return text, meta
                    logger.info(
                        "OCR provider %s returned %d chars (< %d), trying next",
                        meta.get("ocr_provider", "unknown"), len(text.strip()), self.min_text_threshold,
                    )
                except Exception as e:
                    logger.warning("OCR provider failed: %s, trying next", e)

            return "", {"ocr_error": "all providers returned insufficient text"}

        except Exception as e:
            logger.error("OCR fallback failed: %s", e)
            return "", {"ocr_error": str(e)}

    async def _ocr_fallback_async(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Async OCR fallback using provider chain."""
        try:
            from llm_utils_parser.ocr import OCRFactory

            chain = OCRFactory.create_chain(self.config)
            if not chain:
                return "", {}

            for ocr in chain:
                try:
                    text, meta = await ocr.aextract_text(file_path)
                    if len(text.strip()) >= self.min_text_threshold:
                        return text, meta
                    logger.info(
                        "OCR provider %s returned %d chars (< %d), trying next",
                        meta.get("ocr_provider", "unknown"), len(text.strip()), self.min_text_threshold,
                    )
                except Exception as e:
                    logger.warning("Async OCR provider failed: %s, trying next", e)

            return "", {"ocr_error": "all providers returned insufficient text"}

        except Exception as e:
            logger.error("Async OCR fallback failed: %s", e)
            return "", {"ocr_error": str(e)}

    # ── Table extraction ───────────────────────────────────────────

    @staticmethod
    def _extract_tables(file_path: Path) -> Tuple[str, int]:
        """Extract tables from PDF using pdfplumber → Markdown format.

        Returns:
            Tuple of (tables_markdown, table_count).
        """
        try:
            import pdfplumber

            tables_md = []
            with pdfplumber.open(str(file_path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        if not table or not table[0]:
                            continue

                        # Convert to Markdown table
                        md_lines = []
                        # Header
                        header = [str(cell or "") for cell in table[0]]
                        md_lines.append("| " + " | ".join(header) + " |")
                        md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                        # Rows
                        for row in table[1:]:
                            cells = [str(cell or "") for cell in row]
                            # Pad if fewer cells than header
                            while len(cells) < len(header):
                                cells.append("")
                            md_lines.append("| " + " | ".join(cells[:len(header)]) + " |")

                        tables_md.append(
                            f"\n**Table (Page {page_num + 1}, #{table_idx + 1}):**\n"
                            + "\n".join(md_lines)
                        )

            return "\n\n".join(tables_md), len(tables_md)

        except ImportError:
            logger.debug("pdfplumber not installed, skipping table extraction")
            return "", 0
        except Exception as e:
            logger.warning("Table extraction failed: %s", e)
            return "", 0

    # ── Async override ─────────────────────────────────────────────

    async def aextract(self, file_path: Path, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Async PDF extraction using thread pool for hybrid page-by-page extraction."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.extract(file_path, **kwargs))
