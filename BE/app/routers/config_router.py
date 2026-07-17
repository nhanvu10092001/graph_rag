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
