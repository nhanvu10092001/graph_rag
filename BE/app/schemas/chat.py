"""Schemas for chat endpoints."""

from typing import List, Optional
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatStreamRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    config: Optional[dict] = None
    apiKey: Optional[str] = None
    groupId: Optional[int] = None
