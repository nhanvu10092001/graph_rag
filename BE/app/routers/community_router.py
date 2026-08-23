"""Community Detection & Management API Router."""

import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["community"])


@router.post("/api/community/detect")
async def detect_communities():
    """Run community detection on the current knowledge graph."""
    from app.agent import community_service

    try:
        result = community_service.detect_communities()
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Community detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/community/summarize")
async def generate_summaries():
    """Generate LLM summaries for all detected communities."""
    from app.agent import community_service

    try:
        result = community_service.generate_community_summaries()
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Community summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/community/rebuild")
async def rebuild_communities():
    """Full rebuild: clear old communities, re-detect, and re-summarize."""
    from app.agent import community_service

    try:
        result = community_service.rebuild_communities()
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Community rebuild failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/community/list")
async def list_communities(level: int = 0):
    """List all community summaries at a given hierarchy level."""
    from app.agent import community_service

    try:
        communities = community_service.get_community_summaries(level=level)
        return {"status": "success", "level": level, "count": len(communities), "communities": communities}
    except Exception as e:
        logger.error(f"Failed to list communities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/community/search")
async def search_communities(query: str, top_k: int = 5, level: int = 0):
    """Vector search over community summaries."""
    from app.agent import community_service

    try:
        results = community_service.search_communities_by_vector(query, top_k=top_k, level=level)
        return {"status": "success", "query": query, "results": results}
    except Exception as e:
        logger.error(f"Community search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

