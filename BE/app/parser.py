"""Document text extraction utility delegating to llm_utils_parser package with custom config."""

import os
import tempfile
import logging
import yaml
from pathlib import Path
from llm_utils_parser import ExtractorRegistry

logger = logging.getLogger("BE.parser")


def load_rag_config() -> dict:
    """Loads configuration from rag_config.yaml."""
    config_path = Path(__file__).parent.parent / "rag_config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load rag_config.yaml ({e}). Using default configuration.")
    return {}


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Extracts plain text content from various file formats based on extension using llm_utils_parser with rag_config.yaml config."""
    logger.info(f"Extracting text from: {filename}")
    suffix = os.path.splitext(filename)[1].lower()
    
    # Use NamedTemporaryFile to safely handle temp file creation
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name
        
    try:
        config = load_rag_config()
        registry = ExtractorRegistry(config)
        text, _ = registry.extract(Path(tmp_file_path))
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {e}")
        raise ValueError(f"Failed to parse {filename}: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
