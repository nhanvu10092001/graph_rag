"""FastAPI Application Entry Point."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services.file_storage import init_minio
from app.routers import router

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown events."""
    logger.info("Initializing file storage bucket...")
    try:
        init_minio()
    except Exception as e:
        logger.error(f"Failed to initialize file storage: {e}")
    yield
    logger.info("Shutting down FastAPI application...")


app = FastAPI(
    title="Graph RAG Backend",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration using explicit origins from configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register endpoints from routers
app.include_router(router)
