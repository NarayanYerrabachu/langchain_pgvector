"""Models package."""
from app.models.request import (
    QueryRequest,
    DocumentIngestRequest,
    WebScrapingRequest
)
from app.models.response import (
    QueryResponse,
    IngestResponse,
    HealthResponse
)

__all__ = [
    "QueryRequest",
    "DocumentIngestRequest",
    "WebScrapingRequest",
    "QueryResponse",
    "IngestResponse",
    "HealthResponse"
]