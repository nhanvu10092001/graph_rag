"""FastAPI router endpoints for configuration settings."""

import logging
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings
from app.schemas.config import ConfigStatusResponse, ModelListResponse, ModelInfo

logger = logging.getLogger("BE.routers.config_router")

router = APIRouter()


class VerifyKeyRequest(BaseModel):
    apiKey: Optional[str] = None


@router.get("/api/config", response_model=ConfigStatusResponse)
async def get_config():
    has_key = bool(
        getattr(settings, "anthropic_api_key", None)
        or getattr(settings, "openai_api_key", None)
    )
    return ConfigStatusResponse(hasSystemKey=has_key)


@router.get("/v1/models", response_model=ModelListResponse)
@router.get("/models", response_model=ModelListResponse)
async def list_models():
    current_model = settings.llm_model or "gpt-4o-mini"
    model_ids = list(set([current_model, "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "claude-3-7-sonnet-20250219"]))
    return ModelListResponse(
        data=[ModelInfo(id=mid) for mid in model_ids]
    )


@router.post("/api/verify-key")
async def verify_key(req: VerifyKeyRequest):
    logger.info("Verifying configuration and database connectivity...")
    try:
        settings.validate()
        return {"valid": True, "message": "Server configuration is valid and active!"}
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {"valid": False, "message": f"Configuration error: {str(e)}"}
