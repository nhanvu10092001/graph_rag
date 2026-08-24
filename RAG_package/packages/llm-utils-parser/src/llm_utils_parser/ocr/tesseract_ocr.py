"""Tesseract OCR provider — local, lightweight OCR engine."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from PIL import Image

from . import BaseOCR

logger = logging.getLogger(__name__)


class TesseractOCR(BaseOCR):
    """OCR using Tesseract via pytesseract.

    Config (under extraction.ocr.tesseract):
        lang: "vie+eng"
        tesseract_cmd: "/opt/homebrew/bin/tesseract" (optional path to executable)
        config_options: "" (optional extra config options)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.lang = self.config.get("lang", "vie+eng")
        self.tesseract_cmd = self.config.get("tesseract_cmd")
        self.config_options = self.config.get("config_options", "")
        self.render_scale = float(self.config.get("render_scale", 4.0))

    def _setup_cmd(self):
        """Set pytesseract command if configured."""
        import pytesseract
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def extract_image_bytes(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """Extract text from image bytes using Tesseract."""
        import io
        import pytesseract
        from PIL import Image
        self._setup_cmd()
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            return pytesseract.image_to_string(pil_img, lang=self.lang, config=self.config_options)
        except Exception as e:
            logger.warning(f"Tesseract OCR error on image bytes: {e}")
            return ""

    def extract_text(self, file_path: str | Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text using Tesseract OCR.
        
        Supports PDF files (via page rendering with pypdfium2) and image files.
        """
        import pytesseract
        self._setup_cmd()
        
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        t0 = time.perf_counter()
        
        text_content = []
        page_count = 0
        
        if ext == ".pdf":
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(file_path)
                page_count = len(pdf)
                for page_num, page in enumerate(pdf):
                    # Render page at configured scale for OCR accuracy
                    bitmap = page.render(scale=self.render_scale)
                    pil_img = bitmap.to_pil()
                    page_text = pytesseract.image_to_string(
                        pil_img, lang=self.lang, config=self.config_options
                    )
                    if page_text.strip():
                        text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
            except Exception as e:
                logger.error("Tesseract PDF rendering/OCR failed: %s", e)
                raise e
        else:
            # Assume it's an image
            try:
                with Image.open(file_path) as img:
                    page_text = pytesseract.image_to_string(
                        img, lang=self.lang, config=self.config_options
                    )
                    text_content.append(page_text)
                    page_count = 1
            except Exception as e:
                logger.error("Tesseract image OCR failed: %s", e)
                raise e
                
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        content = "\n\n".join(text_content)
        
        metadata = {
            "ocr_provider": "tesseract",
            "ocr_time_ms": elapsed_ms,
            "pages": page_count,
        }
        
        logger.info(
            "tesseract: extracted %d chars in %.1fms from '%s'",
            len(content), elapsed_ms, file_path.name,
        )
        
        return content, metadata
