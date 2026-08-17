"""Schemas for config endpoints."""

from typing import List
from pydantic import BaseModel


class ConfigStatusResponse(BaseModel):
    hasSystemKey: bool


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "system"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]
