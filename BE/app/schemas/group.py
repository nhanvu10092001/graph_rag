"""Schemas for group endpoints."""

from pydantic import BaseModel, Field


class GroupCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class GroupResponse(BaseModel):
    id: int
    name: str
    created_at: str


class GroupDeleteResponse(BaseModel):
    status: str
    message: str
