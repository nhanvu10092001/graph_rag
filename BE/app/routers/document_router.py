"""FastAPI router endpoints for document uploads and listings."""

import logging
from fastapi import APIRouter, UploadFile, File, BackgroundTasks

from app.services.file_storage import upload_file_to_minio
from app.services.document_service import (
    save_document_metadata,
    list_documents,
    index_document_background
)

logger = logging.getLogger("BE.routers.document_router")

router = APIRouter()


@router.get("/api/documents")
async def get_documents():
    """Returns list of uploaded documents and indexing metadata from Postgres."""
    logger.info("Listing documents...")
    try:
        return list_documents()
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        return []


@router.post("/api/documents/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Uploads document to MinIO, saves metadata to Postgres, and schedules background RAG indexing."""
    filename = file.filename
    logger.info(f"Received file upload request: {filename}")
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        # 1. Upload to MinIO
        minio_key = upload_file_to_minio(file_bytes, filename)
        
        # 2. Create document record in Postgres (status="pending")
        doc_metadata = save_document_metadata(filename, minio_key)
        
        # 3. Enqueue background indexing task
        background_tasks.add_task(
            index_document_background, 
            doc_metadata["id"], 
            minio_key, 
            filename
        )
        
        return {
            "status": "success",
            "message": "File uploaded successfully. Indexing started in background.",
            "document": doc_metadata
        }
    except Exception as e:
        logger.error(f"Upload failed for '{filename}': {e}")
        return {
            "status": "error",
            "message": f"Failed to upload document: {str(e)}"
        }
