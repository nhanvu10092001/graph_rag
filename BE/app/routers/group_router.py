"""FastAPI router endpoints for managing article/document groups."""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas.group import GroupCreate, GroupResponse, GroupDeleteResponse
from app.services.document_service import create_group, list_groups, delete_group

logger = logging.getLogger("BE.routers.group_router")

router = APIRouter()


@router.get("/api/groups", response_model=List[GroupResponse])
async def get_groups(db: Session = Depends(get_db)):
    logger.info("Listing document groups...")
    return list_groups(db=db)


@router.post("/api/groups", response_model=GroupResponse)
async def create_new_group(payload: GroupCreate, db: Session = Depends(get_db)):
    logger.info(f"Creating new document group: {payload.name}")
    try:
        return create_group(payload.name, db=db)
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))


@router.delete("/api/groups/{group_id}", response_model=GroupDeleteResponse)
async def delete_existing_group(group_id: int, db: Session = Depends(get_db)):
    logger.info(f"Request to delete group: {group_id}")
    try:
        return delete_group(group_id, db=db)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
