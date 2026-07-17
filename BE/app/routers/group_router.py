"""FastAPI router endpoints for managing article/document groups."""

import logging
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.document_service import (
    create_group,
    list_groups,
    delete_group
)

logger = logging.getLogger("BE.routers.group_router")

router = APIRouter()


class GroupCreate(BaseModel):
    name: str


@router.get("/api/groups")
async def get_groups():
    """Lists all document groups."""
    logger.info("Listing document groups...")
    try:
        return list_groups()
    except Exception as e:
        logger.error(f"Failed to list groups: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list groups: {str(e)}")


@router.post("/api/groups")
async def create_new_group(payload: GroupCreate):
    """Creates a new document group."""
    logger.info(f"Creating new document group: {payload.name}")
    try:
        return create_group(payload.name)
    except ValueError as ve:
        logger.warning(f"Failed to create group: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to create group: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create group: {str(e)}")


@router.delete("/api/groups/{group_id}")
async def delete_existing_group(group_id: int):
    """Deletes group and all associated documents cascade-style."""
    logger.info(f"Request to delete group: {group_id}")
    try:
        return delete_group(group_id)
    except ValueError as ve:
        logger.warning(f"Failed to delete group: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to delete group {group_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete group: {str(e)}")
