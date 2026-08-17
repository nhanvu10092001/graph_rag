"""OCR providers for scanned document text extraction.

Supports:
- Tesseract — local, lightweight OCR engine
- Docling (IBM) — best for complex layouts, tables, formulas
- PaddleOCR — multilingual, excellent Vietnamese support
- DeepSeek OCR — LLM-based OCR via vision model API

Usage:
    from llm_utils_parser.ocr import OCRFactory
    ocr = OCRFactory.create(config)
    text, meta = ocr.extract_text("scan.pdf")
    text, meta = await ocr.aextract_text("scan.pdf")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseOCR(ABC):
    """Abstract base for OCR providers."""

    @abstractmethod
    def extract_text(self, file_path: str | Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text from an image or scanned PDF.

        Args:
            file_path: Path to file.

        Returns:
            Tuple of (extracted_text, ocr_metadata).
        """
        ...

    async def aextract_text(self, file_path: str | Path) -> Tuple[str, Dict[str, Any]]:
        """Async version — default runs sync in thread pool."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.extract_text(file_path)
        )


class OCRFactory:
    """Factory for creating OCR provider instances.

    Config (under extraction.ocr):
        enabled: true
        provider: "tesseract"  # Options: "tesseract", "docling", "paddle", "deepseek"
        tesseract:
            lang: "vie+eng"
        docling:
            ...
        paddle:
            lang: ["vi", "en"]
        deepseek:
            model: "deepseek-chat"
            api_key: "..."
    """

    _providers: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, provider_class: type):
        cls._providers[name.lower()] = provider_class

    @classmethod
    def create(cls, config: Dict[str, Any]) -> Optional[BaseOCR]:
        """Create an OCR provider from config.

        Args:
            config: Full config dict (reads extraction.ocr section).

        Returns:
            OCR provider instance, or None if OCR is disabled.
        """
        ocr_config = {}
        if isinstance(config, dict):
            ocr_config = config.get("extraction", {}).get("ocr", {}) if isinstance(config.get("extraction"), dict) else {}
            if not ocr_config:
                ocr_config = config.get("ocr", {})
            if not ocr_config and "pdf" in config:
                ocr_config = config.get("pdf", {})
            
        enabled = ocr_config.get("enabled", False) or ocr_config.get("enable_ocr", False)
        if not enabled:
            return None

        provider_name = ocr_config.get("provider", "tesseract").lower()
        provider_cfg = ocr_config.get(provider_name, {})

        # Auto-register built-in providers
        if not cls._providers:
            cls._register_defaults()

        provider_class = cls._providers.get(provider_name)
        if provider_class is None:
            logger.error("Unknown OCR provider: %s", provider_name)
            return None

        try:
            return provider_class(provider_cfg)
        except Exception as e:
            logger.error("Failed to create OCR provider '%s': %s", provider_name, e)
            return None

    @classmethod
    def create_chain(cls, config: Dict[str, Any]) -> List[BaseOCR]:
        """Create an ordered list of OCR providers from config fallback_chain.

        Args:
            config: Full config dict (reads extraction.ocr.fallback_chain).

        Returns:
            List of initialized OCR provider instances.
        """
        ocr_config = {}
        if isinstance(config, dict):
            ocr_config = config.get("extraction", {}).get("ocr", {}) if isinstance(config.get("extraction"), dict) else {}
            if not ocr_config:
                ocr_config = config.get("ocr", {})

        chain_names = ocr_config.get("fallback_chain", [ocr_config.get("provider", "tesseract")])

        if not cls._providers:
            cls._register_defaults()

        chain: List[BaseOCR] = []
        for name in chain_names:
            name = name.lower()
            provider_class = cls._providers.get(name)
            if provider_class is None:
                logger.warning("Unknown OCR provider in chain: %s, skipping", name)
                continue
            try:
                provider_cfg = ocr_config.get(name, {})
                chain.append(provider_class(provider_cfg))
            except Exception as e:
                logger.warning("Failed to init OCR provider '%s': %s, skipping", name, e)
        return chain

    @classmethod
    def _register_defaults(cls):
        """Register built-in OCR providers."""
        try:
            from .tesseract_ocr import TesseractOCR
            cls.register("tesseract", TesseractOCR)
        except ImportError:
            pass

        try:
            from .docling_ocr import DoclingOCR
            cls.register("docling", DoclingOCR)
        except ImportError:
            pass

        try:
            from .paddle_ocr import PaddleOCRProvider
            cls.register("paddle", PaddleOCRProvider)
        except ImportError:
            pass

        try:
            from .deepseek_ocr import DeepSeekOCR
            cls.register("deepseek", DeepSeekOCR)
        except ImportError:
            pass
