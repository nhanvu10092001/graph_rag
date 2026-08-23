"""FastAPI router endpoints for document uploads and listings."""

import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Form, Query, Depends, HTTPException
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas.document import DocumentResponse, DocumentUploadResponse, DocumentDeleteResponse
from app.services.file_storage import upload_file_to_minio
from app.services.document_service import (
    save_document_metadata,
    list_documents,
    index_document_background,
    delete_document,
)

logger = logging.getLogger("BE.routers.document_router")

router = APIRouter()


@router.get("/api/documents", response_model=List[DocumentResponse])
async def get_documents(db: Session = Depends(get_db)):
    logger.info("Listing all documents...")
    return list_documents(db=db)


MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


@router.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    filename = file.filename
    logger.info(f"Received file upload request: {filename}")

    if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail=f"File extension not allowed. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File size exceeds {MAX_FILE_SIZE / (1024 * 1024):.0f}MB limit.")

    minio_key = upload_file_to_minio(file_bytes, filename)
    doc_metadata = save_document_metadata(filename, minio_key, db=db)

    background_tasks.add_task(
        index_document_background,
        doc_metadata["id"],
        minio_key,
        filename,
    )

    return {
        "status": "success",
        "message": "File uploaded successfully. Indexing started in background.",
        "document": doc_metadata,
    }


@router.delete("/api/documents/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_uploaded_document(doc_id: int, db: Session = Depends(get_db)):
    logger.info(f"Received request to delete document: {doc_id}")
    try:
        return delete_document(doc_id, db=db)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
