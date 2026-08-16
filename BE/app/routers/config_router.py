"""FastAPI router endpoints for configuration settings."""

import logging
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger("BE.routers.config_router")

router = APIRouter()


class VerifyKeyRequest(BaseModel):
    apiKey: Optional[str] = None


@router.get("/api/config")
async def get_config():
    """Endpoint checked by FE to verify environment configuration status."""
    logger.info("Config status checked.")
    return {
        "hasSystemKey": True  # Since keys are preconfigured in config.yaml / .env
    }


@router.get("/v1/models")
@router.get("/models")
async def list_models():
    """OpenAI compatibility endpoint listing available models."""
    logger.info("Listing available models.")
    current_model = settings.llm_model or "gpt-4o-mini"
    available_models = list(set([current_model, "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "claude-3-5-sonnet-20241022"]))
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1700000000,
                "owned_by": "system"
            }
            for model_id in available_models
        ]
    }


@router.post("/api/verify-key")
async def verify_key(req: VerifyKeyRequest):
    """Endpoint checked by FE settings to verify API key or server status."""
    logger.info("Verifying configuration and database connectivity...")
    try:
        # Check settings validation
        settings.validate()
        return {"valid": True, "message": "Server configuration is valid and active!"}
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {"valid": False, "message": f"Configuration error: {str(e)}"}
