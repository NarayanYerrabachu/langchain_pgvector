"""Query endpoint."""
from typing import Optional

from fastapi import APIRouter, HTTPException
from app.models.request import QueryRequest
from app.models.response import QueryResponse
from app.utils.logger import get_logger
from app.rag.pipeline import RagPipeline
logger = get_logger(__name__)
router = APIRouter()

# Pipeline instance (injected by main.py)
pipeline: Optional[RagPipeline] = None


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.

    Args:
        request: QueryRequest with question and top_k

    Returns:
        QueryResponse with answer and sources
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        logger.info(f"❓ Processing query: {request.question}")
        result = await pipeline.query(request.question, request.top_k)

        return QueryResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))