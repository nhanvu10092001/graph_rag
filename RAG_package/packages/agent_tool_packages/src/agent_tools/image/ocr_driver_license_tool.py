"""OcrDriverLicenseTool — OCR for driver licenses via FastAPI.

Calls:
    POST /ocr/dl_ocr/             → dispatch DL OCR job
    GET  /ocr/get_dl_ocr/{image_id}  → poll for OCR result

The FastAPI endpoint handles MQ dispatch; consumer processes
Azure Form Recognizer with custom model and updates MongoDB.
"""

from typing import Any

from pydantic import BaseModel, Field

from ..abs_interface.api_tool import AbstractApiTool
from ..core.registry import ServiceToolRegistry


class OcrDriverLicenseInput(BaseModel):
    image_id: str | None = Field(default=None, description="UUID of the uploaded image (from upload_image)")
    image_url: str | None = Field(default=None, description="Direct URL to the driver license image (alternative to image_id)")


@ServiceToolRegistry.register
class OcrDriverLicenseTool(AbstractApiTool):
    """Extract fields from a driver license image via FastAPI."""

    @property
    def name(self) -> str:
        return "ocr_driver_license"

    @property
    def description(self) -> str:
        return (
            "Extract structured fields from a driver license image using OCR. "
            "Supports Japanese driver licenses with custom Azure Form Recognizer model. "
            "Requires image_id (from upload_image) OR a direct image_url (not both). "
            "Processing may take up to 2 minutes. "
            "Returns structured data (name, address, license number, dates, etc.)."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return OcrDriverLicenseInput

    @property
    def api_path(self) -> str:
        return "/ocr/dl_ocr/"

    @property
    def poll_endpoint(self) -> str | None:
        return "/ocr/get_dl_ocr/{image_id}"

    @property
    def poll_timeout(self) -> int:
        return 120

    @property
    def poll_interval(self) -> float:
        return 3.0

    def build_request(
        self,
        image_id: str | None = None,
        image_url: str | None = None,
        **_,
    ) -> dict:
        """Build JSON body matching DLOCRInput schema."""
        body: dict = {
            "image_id": image_id,
            "image_url": image_url,
        }
        return {"json": body}

    def parse_response(self, data: Any) -> Any:
        """Parse driver license OCR result.

        From poll GET: the OCR JSON data directly.
        From dispatch POST: {"message": ..., "image_id": ...}
        """
        if isinstance(data, dict) and data.get("status") == "timeout":
            return {
                "status": "timeout",
                "message": data.get("message", "DL OCR processing timed out."),
            }

        return {
            "status": "completed",
            "ocr_data": data,
            "message": "Driver license OCR completed successfully.",
        }
