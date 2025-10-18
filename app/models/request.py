"""Request schemas using Pydantic."""
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any


class QueryRequest(BaseModel):
    """Request schema for querying the RAG system."""
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the RAG system"
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of source documents to retrieve"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the main benefits?",
                "top_k": 5
            }
        }


class DocumentIngestRequest(BaseModel):
    """Request schema for text ingestion."""
    texts: List[str] = Field(
        ...,
        min_items=1,
        description="List of text documents to ingest"
    )
    metadata: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Optional metadata for each document"
    )
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Size of text chunks"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["Document text 1", "Document text 2"],
                "metadata": [{"source": "doc1"}, {"source": "doc2"}],
                "chunk_size": 1000
            }
        }


class WebScrapingRequest(BaseModel):
    """Request schema for web scraping."""
    urls: List[HttpUrl] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="URLs to scrape"
    )
    include_links: bool = Field(
        False,
        description="Include all links in the content"
    )
    timeout: int = Field(
        default=10,
        ge=5,
        le=60,
        description="Request timeout in seconds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "urls": ["https://example.com"],
                "include_links": True,
                "timeout": 15
            }
        }