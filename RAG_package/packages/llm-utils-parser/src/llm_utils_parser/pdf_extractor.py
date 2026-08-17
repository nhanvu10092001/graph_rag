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
    """Extract text from PDF files with OCR fallback.

    Config (under extraction.pdf):
        primary_loader: "pypdf"      # Options: "pypdf", "unstructured"
        enable_ocr: true             # Enable OCR fallback for scanned PDFs
        extract_tables: false        # Extract tables via pdfplumber
        min_text_threshold: 50       # Min chars before triggering OCR
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        pdf_cfg = self.config.get("pdf", {})
        if not pdf_cfg and "extraction" in self.config:
            pdf_cfg = self.config.get("extraction", {}).get("pdf", {})
        self.primary_loader = pdf_cfg.get("primary_loader", "pypdf")
        self.enable_ocr = pdf_cfg.get("enable_ocr", False)
        self.extract_tables = pdf_cfg.get("extract_tables", False)
        self.min_text_threshold = pdf_cfg.get("min_text_threshold", MIN_TEXT_THRESHOLD)


    def extract(self, file_path: Path, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Extract text from PDF with OCR fallback chain."""
        t0 = time.perf_counter()

        metadata = {
            "extractor": "pdf",
            "file_size": file_path.stat().st_size,
        }

        # Step 1: Try standard text extraction
        text, loader_name, pages = self._extract_text_pdf(file_path)
        metadata["loader"] = loader_name
        metadata["pages"] = pages

        # Step 2: OCR fallback if text is too short (likely scanned PDF)
        if self.enable_ocr and len(text.strip()) < self.min_text_threshold:
            logger.info(
                "PDF text too short (%d chars < %d threshold), triggering OCR fallback",
                len(text.strip()), self.min_text_threshold,
            )
            ocr_text, ocr_meta = self._ocr_fallback(file_path)
            if ocr_text:
                text = ocr_text
                metadata.update(ocr_meta)
                metadata["ocr_used"] = True
            else:
                metadata["ocr_used"] = False
                logger.warning("OCR fallback produced no text")
        else:
            metadata["ocr_used"] = False

        # Step 3: Table extraction (optional)
        if self.extract_tables:
            tables_md, table_count = self._extract_tables(file_path)
            if tables_md:
                text = text + "\n\n" + tables_md
                metadata["tables_found"] = table_count

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        metadata["extraction_time_ms"] = elapsed_ms

        logger.info(
            "pdf_extractor: %d chars, loader=%s, ocr=%s, tables=%d in %.1fms",
            len(text), metadata["loader"],
            metadata.get("ocr_provider", "none"),
            metadata.get("tables_found", 0), elapsed_ms,
        )

        return text, metadata

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
        """Async PDF extraction with async OCR fallback."""
        import asyncio

        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()

        metadata = {
            "extractor": "pdf",
            "file_size": file_path.stat().st_size,
        }

        # Text extraction is sync (file I/O), run in thread pool
        text, loader_name, pages = await loop.run_in_executor(
            None, lambda: self._extract_text_pdf(file_path)
        )
        metadata["loader"] = loader_name
        metadata["pages"] = pages

        # OCR fallback (async)
        if self.enable_ocr and len(text.strip()) < self.min_text_threshold:
            ocr_text, ocr_meta = await self._ocr_fallback_async(file_path)
            if ocr_text:
                text = ocr_text
                metadata.update(ocr_meta)
                metadata["ocr_used"] = True
            else:
                metadata["ocr_used"] = False
        else:
            metadata["ocr_used"] = False

        # Table extraction (sync, run in thread pool)
        if self.extract_tables:
            tables_md, table_count = await loop.run_in_executor(
                None, lambda: self._extract_tables(file_path)
            )
            if tables_md:
                text = text + "\n\n" + tables_md
                metadata["tables_found"] = table_count

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        metadata["extraction_time_ms"] = elapsed_ms

        return text, metadata
