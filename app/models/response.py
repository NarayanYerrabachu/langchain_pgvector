"""Response schemas using Pydantic."""
from pydantic import BaseModel
from typing import List, Dict, Any

class QueryResponse(BaseModel):
    """Response schema for query results."""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    timestamp: str

class IngestResponse(BaseModel):
    """Response schema for ingestion results."""
    ingested_chunks: int
    source_count: int
    status: str
    timestamp: str

class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    timestamp: str
    embedding_model: str