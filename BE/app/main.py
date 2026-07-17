"""FastAPI Application Entry Point."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.file_storage import init_minio
from app.routers import router

logger = logging.getLogger("app.main")

app = FastAPI(title="Graph RAG Backend", version="1.0.0")


@app.on_event("startup")
def startup_event():
    logger.info("Initializing file storage bucket...")
    try:
        init_minio()
    except Exception as e:
        logger.error(f"Failed to initialize file storage: {e}")


# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register endpoints from routers
app.include_router(router)
