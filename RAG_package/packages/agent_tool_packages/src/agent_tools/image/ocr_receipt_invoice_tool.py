"""OcrReceiptInvoiceTool — OCR for receipts and invoices via FastAPI.

Calls:
    POST /ocr/receipt_invoice_ocr/  → dispatch OCR job
    GET  /ocr/get_re_inv_ocr/{image_id}  → poll for OCR result

The FastAPI endpoint handles MQ dispatch; consumer processes
YOLO classification + Azure Document Intelligence OCR and updates MongoDB.
"""

from typing import Any

from pydantic import BaseModel, Field

from ..abs_interface.api_tool import AbstractApiTool
from ..core.registry import ServiceToolRegistry


class OcrReceiptInvoiceInput(BaseModel):
    image_id: str | None = Field(default=None, description="UUID of the uploaded image (from upload_image)")
    image_url: str | None = Field(default=None, description="Direct URL to the image (alternative to image_id)")
    language: str = Field(default="Japanese", description="Language of the document for OCR")


@ServiceToolRegistry.register
class OcrReceiptInvoiceTool(AbstractApiTool):
    """Extract text and fields from receipt or invoice images via FastAPI."""

    @property
    def name(self) -> str:
        return "ocr_receipt_invoice"

    @property
    def description(self) -> str:
        return (
            "Extract text and structured fields from a receipt or invoice image using OCR. "
            "The system automatically classifies whether the image is a receipt or invoice. "
            "Requires image_id (from upload_image) OR a direct image_url (not both). "
            "Processing may take up to 2 minutes. "
            "Returns structured OCR data (merchant, items, amounts, dates, etc.)."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return OcrReceiptInvoiceInput

    @property
    def api_path(self) -> str:
        return "/ocr/receipt_invoice_ocr/"

    @property
    def poll_endpoint(self) -> str | None:
        return "/ocr/get_re_inv_ocr/{image_id}"

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
        language: str = "Japanese",
        **_,
    ) -> dict:
        """Build JSON body matching REINVOCRInput schema."""
        body: dict = {
            "language": language,
            "image_id": image_id,
            "image_url": image_url,
        }
        return {"json": body}

    def parse_response(self, data: Any) -> Any:
        """Parse OCR result.

        From poll GET: the OCR JSON data directly.
        From dispatch POST: {"message": ..., "image_id": ...}
        """
        if isinstance(data, dict) and data.get("status") == "timeout":
            return {
                "status": "timeout",
                "message": data.get("message", "OCR processing timed out."),
            }

        return {
            "status": "completed",
            "ocr_data": data,
            "message": "Receipt/invoice OCR completed successfully.",
        }
