"""FastAPI bootstrapper script."""

import logging
import os
import uvicorn
from app.config import settings
from app.main import app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BE.main")

if __name__ == "__main__":
    logger.info(f"Starting FastAPI server via bootstrapper on {settings.server_host}:{settings.server_port}")
    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True
    )
