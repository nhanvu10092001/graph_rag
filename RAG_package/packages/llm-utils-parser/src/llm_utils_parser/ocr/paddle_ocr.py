"""PaddleOCR provider — multilingual OCR with excellent Vietnamese support.
Updated to use the official OpenAI Python SDK client interface (OpenAI-compatible vision API).
"""

import base64
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import BaseOCR

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Extract all text from this image."


class PaddleOCRProvider(BaseOCR):
    """OCR using an external OpenAI-compatible Paddle OCR-VL API via OpenAI SDK client.

    Config (under extraction.ocr.paddle):
        base_url / api_url: "http://paddleocr:8000/v1"
        api_key: "EMPTY"
        model: "PaddlePaddle/PaddleOCR-VL-1.5"
        lang: "vi"
        prompt: "Extract all text from this image."
        timeout: 120.0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Resolve base_url from config (support api_url or base_url)
        raw_url = self.config.get("base_url") or self.config.get("api_url", "http://paddleocr:8000/v1")
        if raw_url.endswith("/chat/completions"):
            raw_url = raw_url[:-len("/chat/completions")]
        elif raw_url.endswith("/ocr"):
            raw_url = raw_url[:-len("/ocr")]

        self.base_url = raw_url.rstrip("/")
        self.api_key = self.config.get("api_key", "EMPTY")
        self.model = self.config.get("model", "PaddlePaddle/PaddleOCR-VL-1.5")
        self.lang = self.config.get("lang", "vi")
        self.prompt = self.config.get("prompt", DEFAULT_PROMPT)
        self.timeout = float(self.config.get("timeout", 120.0))
        self._client = None

    def _get_client(self):
        """Lazy-init OpenAI-compatible client."""
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "EMPTY",
            )
            return self._client
        except ImportError:
            raise ImportError(
                "OpenAI SDK is required for Paddle OCR-VL provider. "
                "Install with: pip install openai"
            )

    def extract_image_bytes(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """Extract text directly from in-memory image bytes using OpenAI-compatible Vision API."""
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
                timeout=self.timeout,
            )

            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"Paddle OCR error extracting from image bytes: {e}")
        return ""

    def _get_image_data_list(self, file_path: Path) -> List[str]:
        """Convert a file (image or PDF) into a list of base64 data URIs."""
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            images_base64 = []

            # Use pypdfium2 for robust PDF rendering
            try:
                import io
                import pypdfium2 as pdfium

                pdf = pdfium.PdfDocument(str(file_path))
                for page_num in range(len(pdf)):
                    try:
                        page = pdf[page_num]
                        # scale=2.0 roughly corresponds to 144 DPI
                        bitmap = page.render(scale=2.0)
                        pil_image = bitmap.to_pil()

                        buf = io.BytesIO()
                        pil_image.save(buf, format="PNG")
                        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
                        images_base64.append(f"data:image/png;base64,{encoded}")
                    except Exception as e:
                        logger.warning(f"Failed to render page {page_num} of {file_path} with pdfium: {e}")
                return images_base64
            except ImportError:
                logger.debug("pypdfium2 not found, falling back to pymupdf")

            # Fallback to PyMuPDF if pypdfium2 is unavailable
            try:
                import pymupdf as fitz
            except ImportError:
                try:
                    import fitz  # Fallback for older versions
                except ImportError:
                    raise ImportError("pypdfium2 or pymupdf is required for OCR on PDFs.")

            doc = fitz.open(str(file_path))
            for page_num, page in enumerate(doc):
                try:
                    pix = page.get_pixmap(dpi=150)
                    img_data = pix.tobytes("png")
                    encoded = base64.b64encode(img_data).decode('utf-8')
                    images_base64.append(f"data:image/png;base64,{encoded}")
                except Exception as e:
                    logger.warning(f"Failed to render page {page_num} of {file_path}: {e}")
            doc.close()
            return images_base64

        else:
            # Handle regular images
            with open(file_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type = "image/png" if ext == ".png" else "image/jpeg"
            return [f"data:{mime_type};base64,{encoded}"]

    def extract_text(self, file_path: str | Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text from images or scanned PDFs using OpenAI SDK client."""
        file_path = Path(file_path)
        t0 = time.perf_counter()

        all_content = []
        total_detections = 0
        avg_confidence = 1.0

        try:
            client = self._get_client()
            images_base64 = self._get_image_data_list(file_path)
            num_pages = len(images_base64)

            for image_url in images_base64:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url,
                                    },
                                },
                            ],
                        }
                    ],
                    timeout=self.timeout,
                )

                if response.choices and len(response.choices) > 0:
                    page_content = response.choices[0].message.content or ""
                    if page_content:
                        all_content.append(page_content)
                        total_detections += 1
                else:
                    logger.warning(f"Unexpected response structure from Paddle OCR-VL: {response}")

        except ImportError as e:
            logger.error(f"Dependency error: {e}")
            return "", {"ocr_error": str(e)}
        except Exception as e:
            logger.error(f"Error communicating with Paddle OCR-VL via OpenAI client at {self.base_url}: {e}")
            return "", {"ocr_error": str(e)}

        content = "\n\n".join(all_content)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        metadata = {
            "ocr_provider": "paddle_api",
            "ocr_time_ms": elapsed_ms,
            "lang": self.lang,
            "pages": num_pages if 'num_pages' in locals() else 0,
            "detections": total_detections,
            "avg_confidence": avg_confidence,
            "base_url": self.base_url,
            "model": self.model,
        }

        logger.info(
            "paddle_ocr_api: extracted %d chars from %d pages in %.1fms",
            len(content), num_pages if 'num_pages' in locals() else 0, elapsed_ms,
        )

        return content, metadata
