"""DeepSeek OCR provider — LLM-based OCR via vision model API."""

import base64
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import BaseOCR

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Extract ALL text content from this document image. "
    "Preserve the original structure including headings, paragraphs, "
    "lists, and tables (format tables as Markdown). "
    "If there are charts or figures, describe their content briefly. "
    "Return ONLY the extracted text, no commentary."
)


class DeepSeekOCR(BaseOCR):
    """OCR using DeepSeek VL (vision-language model) API.

    Config (under extraction.ocr.deepseek):
        model: "deepseek-chat"
        api_key: "sk-..."
        base_url: "https://api.deepseek.com"
        prompt: null  # Custom extraction prompt
        max_tokens: 4096
        max_pages: 20  # Max pages to process (for PDFs)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = self.config.get("model", "deepseek-chat")
        self.api_key = self.config.get("api_key", "")
        self.base_url = self.config.get("base_url", "https://api.deepseek.com")
        self.prompt = self.config.get("prompt", DEFAULT_PROMPT)
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.max_pages = self.config.get("max_pages", 20)
        self._client = None

    def _get_client(self):
        """Lazy-init OpenAI-compatible client for DeepSeek API."""
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI

            if not self.api_key:
                raise ValueError(
                    "DeepSeek API key is required. Set extraction.ocr.deepseek.api_key in config."
                )

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            return self._client

        except ImportError:
            raise ImportError(
                "OpenAI SDK is required for DeepSeek OCR. "
                "Install with: pip install openai"
            )

    def extract_image_bytes(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """Extract text from in-memory image bytes using DeepSeek vision model API."""
        try:
            client = self._get_client()
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            image_url = f"data:{mime_type};base64,{encoded}"

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
            )

            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"DeepSeek OCR error on image bytes: {e}")
        return ""

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _pdf_to_images(self, pdf_path: Path) -> list:
        """Convert PDF pages to images for VLM processing."""
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(
                str(pdf_path),
                dpi=200,
                first_page=1,
                last_page=self.max_pages,
            )
            return images

        except ImportError:
            raise ImportError(
                "pdf2image is required for PDF → image conversion. "
                "Install with: pip install pdf2image. "
                "Also requires poppler: apt-get install poppler-utils"
            )

    def _ocr_single_image(self, image_data: str, page_num: int = 1) -> str:
        """Send a single image to DeepSeek VL for OCR."""
        client = self._get_client()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content or ""

    def extract_text(self, file_path: str | Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text from images or scanned PDFs using DeepSeek VL."""
        import io

        file_path = Path(file_path)
        t0 = time.perf_counter()
        pages_processed = 0

        ext = file_path.suffix.lower()

        if ext == ".pdf":
            # Convert PDF to images, then OCR each page
            images = self._pdf_to_images(file_path)
            page_texts = []

            for i, img in enumerate(images):
                # Convert PIL Image to base64
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                page_text = self._ocr_single_image(img_b64, page_num=i + 1)
                page_texts.append(page_text)
                pages_processed += 1

                logger.debug("DeepSeek OCR: page %d/%d done", i + 1, len(images))

            content = "\n\n".join(page_texts)

        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
            # Direct image OCR
            img_b64 = self._encode_image(file_path)
            content = self._ocr_single_image(img_b64)
            pages_processed = 1

        else:
            raise ValueError(f"DeepSeek OCR doesn't support extension: {ext}")

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        metadata = {
            "ocr_provider": "deepseek",
            "ocr_time_ms": elapsed_ms,
            "model": self.model,
            "pages_processed": pages_processed,
        }

        logger.info(
            "deepseek_ocr: extracted %d chars from %d pages in %.1fms",
            len(content), pages_processed, elapsed_ms,
        )

        return content, metadata

    async def aextract_text(self, file_path: str | Path) -> Tuple[str, Dict[str, Any]]:
        """Async extraction using DeepSeek API."""
        import io

        file_path = Path(file_path)
        t0 = time.perf_counter()

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except ImportError:
            # Fallback to sync
            return await super().aextract_text(file_path)

        ext = file_path.suffix.lower()
        pages_processed = 0

        if ext == ".pdf":
            images = self._pdf_to_images(file_path)
            page_texts = []

            for i, img in enumerate(images):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_b64}",
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=self.max_tokens,
                )
                page_texts.append(response.choices[0].message.content or "")
                pages_processed += 1

            content = "\n\n".join(page_texts)

        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
            img_b64 = self._encode_image(file_path)

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}",
                                },
                            },
                        ],
                    }
                    ],
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content or ""
            pages_processed = 1
        else:
            raise ValueError(f"DeepSeek OCR doesn't support extension: {ext}")

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        metadata = {
            "ocr_provider": "deepseek",
            "ocr_time_ms": elapsed_ms,
            "model": self.model,
            "pages_processed": pages_processed,
        }

        return content, metadata
