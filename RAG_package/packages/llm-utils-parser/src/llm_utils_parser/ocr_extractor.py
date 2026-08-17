"""OCR Extractor — adapter wrapping OCR providers as a standard BaseExtractor."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from llm_utils_parser.base import BaseExtractor

logger = logging.getLogger(__name__)


class OCRExtractor(BaseExtractor):
    """Adapter that exposes OCR providers through the BaseExtractor interface.

    Uses OCRFactory to create a provider chain from config, then delegates
    to each provider in order until one returns non-empty text.

    Config (under extraction.ocr):
        provider: "paddle"
        fallback_chain: ["paddle", "deepseek", "tesseract"]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def extract(self, file_path: Path, **kwargs) -> Tuple[str, Dict[str, Any]]:
        from llm_utils_parser.ocr import OCRFactory

        t0 = time.perf_counter()

        chain = OCRFactory.create_chain(self.config)
        if not chain:
            logger.warning("OCR enabled but no providers available")
            return "", {"extractor": "ocr", "error": "no_providers"}

        for ocr in chain:
            try:
                text, meta = ocr.extract_text(file_path)
                if text.strip():
                    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
                    meta["extractor"] = "ocr"
                    meta["extraction_time_ms"] = elapsed_ms
                    logger.info(
                        "ocr_extractor: %d chars via %s in %.1fms",
                        len(text), meta.get("ocr_provider", "unknown"), elapsed_ms,
                    )
                    return text, meta
                logger.debug(
                    "OCR provider %s returned empty text, trying next",
                    meta.get("ocr_provider", "unknown"),
                )
            except Exception as e:
                logger.warning("OCR provider failed: %s, trying next", e)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return "", {
            "extractor": "ocr",
            "error": "all_providers_empty",
            "extraction_time_ms": elapsed_ms,
        }

    async def aextract(self, file_path: Path, **kwargs) -> Tuple[str, Dict[str, Any]]:
        from llm_utils_parser.ocr import OCRFactory

        t0 = time.perf_counter()

        chain = OCRFactory.create_chain(self.config)
        if not chain:
            return "", {"extractor": "ocr", "error": "no_providers"}

        for ocr in chain:
            try:
                text, meta = await ocr.aextract_text(file_path)
                if text.strip():
                    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
                    meta["extractor"] = "ocr"
                    meta["extraction_time_ms"] = elapsed_ms
                    return text, meta
            except Exception as e:
                logger.warning("Async OCR provider failed: %s, trying next", e)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return "", {
            "extractor": "ocr",
            "error": "all_providers_empty",
            "extraction_time_ms": elapsed_ms,
        }
