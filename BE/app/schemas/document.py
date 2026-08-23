"""Schemas for document endpoints."""

from typing import Optional
from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: int
    filename: str
    minio_key: str
    status: str
    entity_count: Optional[int] = None
    relationship_count: Optional[int] = None
    created_at: str


class DocumentUploadResponse(BaseModel):
    status: str
    message: str
    document: DocumentResponse


class DocumentDeleteResponse(BaseModel):
    status: str
    message: str
