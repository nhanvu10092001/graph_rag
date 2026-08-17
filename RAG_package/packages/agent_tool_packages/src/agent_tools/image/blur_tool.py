"""BlurTool — Blur faces and license plates in an image via FastAPI.

Calls:
    POST /img_blur/image_blur/ with syn=true
    → FastAPI dispatches to Kafka, consumer processes, webhooks to Redis
    → FastAPI polls Redis internally and returns blurred image

After receiving the result, generates a presigned MinIO URL so the user
can download the blurred image.
"""

import os
from typing import Any

from pydantic import BaseModel, Field

from ..abs_interface.api_tool import AbstractApiTool
from ..core.registry import ServiceToolRegistry


class BlurInput(BaseModel):
    image_id: str = Field(description="UUID of the uploaded image (from upload_image tool)")


@ServiceToolRegistry.register
class BlurTool(AbstractApiTool):
    """Blur faces and license plates in an image via FastAPI."""

    @property
    def name(self) -> str:
        return "blur_image"

    @property
    def description(self) -> str:
        return (
            "Blur faces and license plates in an image for privacy protection. "
            "Requires image_id from a previous upload_image call. "
            "Processing takes a few seconds. "
            "Returns a download URL for the blurred image."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return BlurInput

    @property
    def api_path(self) -> str:
        return "/img_blur/image_blur/"

    def build_request(self, image_id: str, **_) -> dict:
        """Build JSON body matching ImageBlurInput schema.

        syn=True so FastAPI waits for the result internally (polls Redis)
        and returns the blurred image directly.
        """
        return {
            "json": {"image_id": image_id, "syn": True},
        }

    async def execute(self, **kwargs) -> Any:
        """Override execute to generate presigned URL after blur completes.

        The consumer saves blurred images at: blur_processed/{image_id}.jpeg
        After the API call succeeds, we generate a presigned MinIO URL for download.
        """
        result = await super().execute(**kwargs)

        # If blur succeeded (binary image returned), generate download URL
        image_id = kwargs.get("image_id", "")
        if isinstance(result, dict) and result.get("status") == "completed" and image_id:
            s3_blurred_path = f"blur_processed/{image_id}.jpeg"
            download_url = self._generate_presigned_url(s3_blurred_path)
            if download_url:
                result["download_url"] = download_url
                result["s3_blurred_path"] = s3_blurred_path
                result["message"] = (
                    f"Image blurred successfully. "
                    f"Download the blurred image here: {download_url}"
                )

        return result

    def parse_response(self, data: Any) -> Any:
        """Parse the blur result."""
        if isinstance(data, dict) and data.get("status") == "timeout":
            return {
                "status": "timeout",
                "message": data.get("message", "Blur processing timed out."),
            }

        return {
            "status": "completed",
            "result": data,
            "message": "Image has been blurred successfully.",
        }

    @staticmethod
    def _generate_presigned_url(s3_path: str) -> str:
        """Generate a presigned MinIO URL for downloading the blurred image."""
        try:
            from ..core.clients import get_s3_client, get_s3_bucket

            s3 = get_s3_client()
            bucket = get_s3_bucket()
            return s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": s3_path},
                ExpiresIn=3600,
            )
        except Exception:
            return ""
