"""Health check endpoint."""
from fastapi import APIRouter, HTTPException
from app.models.response import HealthResponse
from app.utils.logger import get_logger
from datetime import datetime

logger = get_logger(__name__)
router = APIRouter()

# Pipeline instance (injected by main.py)
pipeline = None


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse with status information
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized"
        )

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        embedding_model=pipeline.embedding_model.engine_name
    )