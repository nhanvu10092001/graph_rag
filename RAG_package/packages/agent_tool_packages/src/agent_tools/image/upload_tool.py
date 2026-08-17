"""UploadImageTool — Upload an image to the system via FastAPI.

Calls: POST /io/upload_file (multipart file upload)
The FastAPI endpoint handles S3 upload + MQ dispatch to consumer.
Returns image_id immediately (fire-and-forget on server side).

Supports two input modes:
  - image_url:  download from URL, then upload
  - file_path:  read local file directly (e.g. from Gradio upload)
"""

import os
from io import BytesIO
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

from ..abs_interface.api_tool import AbstractApiTool
from ..core.registry import ServiceToolRegistry


class UploadImageInput(BaseModel):
    image_url: Optional[str] = Field(
        default=None,
        description="URL of the image to download and upload to internal storage.",
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Local file path of the image to upload (e.g. from user upload).",
    )


@ServiceToolRegistry.register
class UploadImageTool(AbstractApiTool):
    """Upload an image (from URL or local file) to internal S3 storage via FastAPI."""

    @property
    def name(self) -> str:
        return "upload_image"

    @property
    def description(self) -> str:
        return (
            "Upload an image to internal storage (S3/MinIO). "
            "Accepts either image_url (remote) or file_path (local file from user upload). "
            "This is typically the FIRST step before using blur_image or OCR tools. "
            "Returns: image_id needed for subsequent processing tools."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return UploadImageInput

    @property
    def api_path(self) -> str:
        return "/io/upload_file"

    @property
    def is_multipart(self) -> bool:
        return True

    def build_request(
        self,
        image_url: str | None = None,
        file_path: str | None = None,
        **_,
    ) -> dict:
        """Build multipart upload request from URL or local file.

        Priority: file_path > image_url.
        The FastAPI endpoint expects: file: UploadFile
        """
        if file_path and os.path.isfile(file_path):
            # ── Local file ──────────────────────────────────────────
            filename = os.path.basename(file_path)
            content_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            return {
                "files": {"file": (filename, BytesIO(file_bytes), content_type)},
            }

        if image_url:
            # ── Remote URL ──────────────────────────────────────────
            try:
                headers = {"User-Agent": "FusiAI-Agent/1.0"}
                resp = httpx.get(image_url, timeout=30, follow_redirects=True, headers=headers)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                raise ValueError(f"Failed to download image from {image_url}: {e}") from e

            filename = image_url.split("/")[-1].split("?")[0] or "image.jpg"
            return {
                "files": {"file": (filename, BytesIO(resp.content), "image/jpeg")},
            }

        raise ValueError("Either image_url or file_path must be provided.")

    def parse_response(self, data: Any) -> Any:
        """Parse upload response: {"image_id": "..."}"""
        return {
            "status": "uploaded",
            "image_id": data.get("image_id", ""),
            "message": "Image uploaded successfully. Use image_id with blur_image or OCR tools.",
        }
