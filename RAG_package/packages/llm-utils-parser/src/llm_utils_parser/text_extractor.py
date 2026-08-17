"""Text file extractor — .txt, .md, .csv, .json, .yaml, etc."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from llm_utils_parser.base import BaseExtractor

logger = logging.getLogger(__name__)

ENCODING_CHAIN = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]


class TextExtractor(BaseExtractor):
    """Extract content from plain text files with encoding detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        enc_cfg = cfg.get("encoding", {})
        if not enc_cfg and "extraction" in cfg:
            enc_cfg = cfg.get("extraction", {}).get("encoding", {})
        self.encodings = enc_cfg.get("preferred", ENCODING_CHAIN)
        self.fallback = enc_cfg.get("fallback_strategy", "ignore")


    def extract(self, file_path: Path, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Extract text with multi-encoding fallback."""
        for encoding in self.encodings:
            try:
                content = file_path.read_text(encoding=encoding)
                return content, {
                    "extractor": "text",
                    "encoding": encoding,
                    "file_size": file_path.stat().st_size,
                }
            except UnicodeDecodeError:
                continue

        # Final fallback
        content = file_path.read_text(encoding="utf-8", errors=self.fallback)
        return content, {
            "extractor": "text",
            "encoding": f"utf-8+{self.fallback}",
            "file_size": file_path.stat().st_size,
        }
