"""Unified HTTP error classes for the application."""

from fastapi import HTTPException


class DocumentNotFoundError(HTTPException):
    def __init__(self, doc_id: int):
        super().__init__(status_code=404, detail=f"Document with ID {doc_id} not found.")
