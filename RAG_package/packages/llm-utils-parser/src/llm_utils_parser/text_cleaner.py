"""Text cleaning and normalization pipeline for extracted content."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TextCleaner:
    """Configurable text cleaning pipeline.

    Runs a sequence of cleaning steps on extracted text before chunking.
    Each step can be enabled/disabled via config.

    Usage:
        cleaner = TextCleaner(config)
        cleaned = cleaner.clean(raw_text, file_type="pdf")

    Config (under extraction.cleaning):
        enabled: true
        remove_null_bytes: true
        normalize_unicode: true
        collapse_whitespace: true
        remove_page_markers: true
        strip_headers_footers: false
        min_line_length: 3
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        # Support both nested extraction.cleaning config and direct config
        cleaning_cfg = cfg.get("extraction", {}).get("cleaning", {}) if "extraction" in cfg else cfg
        
        self.enabled = cleaning_cfg.get("enabled", True)
        self.remove_null_bytes = cleaning_cfg.get("remove_null_bytes", True)
        self.normalize_unicode = cleaning_cfg.get("normalize_unicode", True)
        self.collapse_whitespace = cleaning_cfg.get("collapse_whitespace", True)
        self.remove_page_markers = cleaning_cfg.get("remove_page_markers", True)
        self.strip_headers_footers = cleaning_cfg.get("strip_headers_footers", False)
        self.min_line_length = cleaning_cfg.get("min_line_length", 3)

    def clean(self, text: str, file_type: str = "text") -> str:
        """Run the full cleaning pipeline.

        Args:
            text: Raw extracted text.
            file_type: Source file type for type-specific cleaning.

        Returns:
            Cleaned text string.
        """
        if not self.enabled or not text:
            return text

        original_len = len(text)

        if self.remove_null_bytes:
            text = self._remove_null_bytes(text)

        if self.normalize_unicode:
            text = self._normalize_unicode(text)

        if self.remove_page_markers and file_type == "pdf":
            text = self._remove_page_markers(text)

        if self.strip_headers_footers and file_type == "pdf":
            text = self._strip_repeating_headers(text)

        if self.collapse_whitespace:
            text = self._collapse_whitespace(text)

        text = text.strip()

        cleaned_len = len(text)
        if original_len > 0:
            reduction = round((1 - cleaned_len / original_len) * 100, 1)
            if reduction > 5:
                logger.info(
                    "text_cleaner: %d → %d chars (%.1f%% reduction)",
                    original_len, cleaned_len, reduction,
                )

        return text

    # ── Individual cleaning steps ──────────────────────────────────────

    @staticmethod
    def _remove_null_bytes(text: str) -> str:
        """Remove null bytes, BOM, and zero-width characters."""
        # Null bytes
        text = text.replace("\x00", "")
        # BOM
        text = text.replace("\ufeff", "")
        text = text.replace("\ufffe", "")
        # Zero-width characters
        text = text.replace("\u200b", "")  # zero-width space
        text = text.replace("\u200c", "")  # zero-width non-joiner
        text = text.replace("\u200d", "")  # zero-width joiner
        text = text.replace("\u2060", "")  # word joiner
        return text

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize unicode to NFKC form.

        Converts ligatures (ﬁ→fi), fullwidth chars, etc.
        """
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        """Collapse excessive whitespace while preserving paragraph structure."""
        # Replace tabs with spaces
        text = text.replace("\t", "    ")
        # Collapse multiple spaces into single space (within lines)
        text = re.sub(r"[^\S\n]+", " ", text)
        # Collapse 3+ newlines into 2 (preserve paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove trailing whitespace on each line
        text = re.sub(r" +\n", "\n", text)
        return text

    @staticmethod
    def _remove_page_markers(text: str) -> str:
        """Remove PDF page markers and standalone page numbers."""
        # Remove "--- Page N ---" markers
        text = re.sub(r"---\s*Page\s*\d+\s*---\n?", "", text)
        # Remove standalone page numbers (a line with just a number)
        text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
        # Remove "Page N of M" patterns
        text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _strip_repeating_headers(text: str) -> str:
        """Detect and remove repeating header/footer lines across pages.

        Heuristic: if a short line (<80 chars) appears 3+ times,
        it's likely a header or footer.
        """
        lines = text.split("\n")
        line_counts: Dict[str, int] = {}

        for line in lines:
            stripped = line.strip()
            if 5 < len(stripped) < 80:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

        # Lines appearing 3+ times are likely headers/footers
        repeating = {line for line, count in line_counts.items() if count >= 3}

        if repeating:
            logger.debug("Detected %d repeating header/footer patterns", len(repeating))
            lines = [line for line in lines if line.strip() not in repeating]

        return "\n".join(lines)
