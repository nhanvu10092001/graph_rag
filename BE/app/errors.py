"""Unified HTTP error classes for the application."""

from fastapi import HTTPException


class DocumentNotFoundError(HTTPException):
    def __init__(self, doc_id: int):
        super().__init__(status_code=404, detail=f"Document with ID {doc_id} not found.")


class GroupNotFoundError(HTTPException):
    def __init__(self, group_id: int):
        super().__init__(status_code=404, detail=f"Group with ID {group_id} not found.")


class GroupAlreadyExistsError(HTTPException):
    def __init__(self, name: str):
        super().__init__(status_code=409, detail=f"Group with name '{name}' already exists.")
