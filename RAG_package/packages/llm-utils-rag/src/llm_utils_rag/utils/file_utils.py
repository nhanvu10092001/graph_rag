"""File utilities for checking supported files and extracting content using llm-utils-parser."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, BinaryIO, TextIO

import httpx
from llm_utils_parser import ExtractorRegistry, is_extension_supported
from .models import File

logger = logging.getLogger(__name__)


def is_s3_url(path: str) -> bool:
    """Check if the path is an S3 URL."""
    return isinstance(path, str) and (path.startswith('s3://') or 's3.amazonaws.com' in path or 'amazonaws.com' in path)


def get_filename_from_s3_url(url: str) -> str:
    """Get the filename from an S3 URL."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    path = parsed.path
    filename = os.path.basename(path)
    return filename or 'file.pdf'


def is_supported_file(file_input: Any) -> bool:
    """Check if the file type is supported for extraction."""
    if isinstance(file_input, File):
        filename = file_input.documents_name
        if filename:
            return is_extension_supported(filename)
        elif file_input.documents_path and is_s3_url(file_input.documents_path):
            filename = get_filename_from_s3_url(file_input.documents_path)
            return is_extension_supported(filename)
        return False
        
    elif hasattr(file_input, 'documents_name'):
        filename = getattr(file_input, 'documents_name', None)
        if filename:
            return is_extension_supported(filename)
        documents_path = getattr(file_input, 'documents_path', None)
        if documents_path and is_s3_url(documents_path):
            filename = get_filename_from_s3_url(documents_path)
            return is_extension_supported(filename)
        return False

    elif hasattr(file_input, 'read'):
        if hasattr(file_input, 'name'):
            return is_extension_supported(file_input.name)
        return True

    elif isinstance(file_input, str) and is_s3_url(file_input):
        filename = get_filename_from_s3_url(file_input)
        return is_extension_supported(filename)

    else:
        return is_extension_supported(str(file_input))


def _get_registry() -> ExtractorRegistry:
    """Get registry configured for default PDF extraction (OCR fallback)."""
    config = {
        "pdf": {
            "enable_ocr": True,
            "primary_loader": "pypdf",
            "min_text_threshold": 50,
        },
        "extraction": {
            "ocr": {
                "enabled": True,
                "provider": "tesseract",
                "tesseract": {
                    "lang": "vie+eng",
                }
            }
        }
    }
    return ExtractorRegistry(config)

def extract_text_from_file(file_input: Any, config: Optional[Dict[str, Any]] = None) -> tuple[str, dict]:
    """Extract content from file, S3 URL, database model, or file object using ExtractorRegistry."""
    registry = ExtractorRegistry(config) if config is not None else _get_registry()
    
    
    # 1. Handle database model or File dataclass
    if isinstance(file_input, File) or (hasattr(file_input, 'documents_name') and hasattr(file_input, 'documents_path')):
        if isinstance(file_input, File):
            filename = file_input.documents_name or 'unknown_file'
            file_url = file_input.documents_path
            file_size = 0
            file_type_attr = 'unknown'
        else:
            filename = getattr(file_input, 'documents_name', 'unknown_file')
            file_url = getattr(file_input, 'documents_path', None)
            file_size = getattr(file_input, 'file_size', 0)
            file_type_attr = getattr(file_input, 'file_type', 'unknown')

        metadata = {
            'filename': filename,
            'file_url': file_url,
            'file_size': file_size,
            'file_type_attr': file_type_attr,
            'source_type': 'file_dataclass' if isinstance(file_input, File) else 'database_model'
        }

        # Add extra File dataclass attributes if available
        if isinstance(file_input, File):
            for attr in ['collection_name', 'category', 'openai_models', 'company_api_keys']:
                value = getattr(file_input, attr, None)
                if value is not None:
                    metadata[attr] = value

        if not file_url:
            raise ValueError("File object must have a documents_path attribute")

        if is_s3_url(file_url):
            # Download file content synchronously
            try:
                response = httpx.get(file_url)
                response.raise_for_status()
                content_bytes = response.content
            except Exception as e:
                raise Exception(f"Failed to download S3 file {file_url}: {e}")

            suffix = os.path.splitext(filename)[1].lower() or '.pdf'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_path = temp_file.name
                temp_file.write(content_bytes)

            try:
                content, ext_meta = registry.extract(Path(temp_path))
                metadata.update(ext_meta)
                metadata['source_type'] = 'database_model_s3'
                return content, metadata
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        else:
            # Local file path
            content, ext_meta = registry.extract(Path(file_url))
            metadata.update(ext_meta)
            return content, metadata

    # 2. Handle file objects
    elif hasattr(file_input, 'read'):
        filename = getattr(file_input, 'name', 'unknown_file')
        suffix = os.path.splitext(filename)[1].lower() or '.txt'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            content_bytes = file_input.read()
            if isinstance(content_bytes, str):
                content_bytes = content_bytes.encode('utf-8')
            temp_file.write(content_bytes)

        try:
            content, ext_meta = registry.extract(Path(temp_path))
            metadata = {
                'filename': filename,
                'file_path': filename,
                'file_size': len(content_bytes),
                'source_type': 'file_object',
                **ext_meta
            }
            return content, metadata
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    # 3. Handle URLs
    elif isinstance(file_input, str) and is_s3_url(file_input):
        filename = get_filename_from_s3_url(file_input)
        suffix = os.path.splitext(filename)[1].lower() or '.pdf'
        
        try:
            response = httpx.get(file_input)
            response.raise_for_status()
            content_bytes = response.content
        except Exception as e:
            raise Exception(f"Failed to download URL {file_input}: {e}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            temp_file.write(content_bytes)

        try:
            content, ext_meta = registry.extract(Path(temp_path))
            metadata = {
                'filename': filename,
                'file_path': file_input,
                'file_size': len(content_bytes),
                'source_type': 'url',
                **ext_meta
            }
            return content, metadata
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    # 4. Handle regular local file paths
    else:
        file_path = Path(file_input)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        content, ext_meta = registry.extract(file_path)
        metadata = {
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'source_type': 'local_file',
            **ext_meta
        }
        return content, metadata


async def extract_file_content_async(file_input: Any, config: Optional[Dict[str, Any]] = None) -> tuple[str, dict]:
    """Asynchronously extract content using ExtractorRegistry."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: extract_text_from_file(file_input, config))

