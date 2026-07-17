"""Router package aggregator grouping sub-routers."""

from fastapi import APIRouter
from app.routers.config_router import router as config_router
from app.routers.chat_router import router as chat_router
from app.routers.document_router import router as document_router

router = APIRouter()

# Include sub-routers
router.include_router(config_router)
router.include_router(chat_router)
router.include_router(document_router)
