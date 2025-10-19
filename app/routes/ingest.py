"""Document ingestion endpoints."""
from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List, Optional

from langchain_core.documents import Document

from app.models.request import DocumentIngestRequest, WebScrapingRequest
from app.models.response import IngestResponse
from app.loaders.pdf import PDFProcessor
from app.loaders.web import WebScraper
from app.rag import RagPipeline
from app.utils.logger import get_logger

from pathlib import Path
from datetime import datetime

logger = get_logger(__name__)
router = APIRouter()

# Pipeline instance (injected by main.py)
pipeline: Optional[RagPipeline] = None
TEMP_DIR = Path("./temp_uploads")


@router.post("/texts", response_model=IngestResponse)
async def ingest_texts(request: DocumentIngestRequest):
    """
    Ingest plain text documents.

    Args:
        request: DocumentIngestRequest with texts and optional metadata

    Returns:
        IngestResponse with ingestion status
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        logger.info(f"📝 Ingesting {len(request.texts)} text documents")

        documents = [
            Document(
                page_content=text,
                metadata=(
                    request.metadata[i]
                    if request.metadata and i < len(request.metadata)
                    else {}
                )
            )
            for i, text in enumerate(request.texts)
        ]

        chunks = await pipeline.ingest_documents(documents, request.chunk_size)

        return IngestResponse(
            ingested_chunks=chunks,
            source_count=len(request.texts),
            status="success",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error ingesting texts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pdf", response_model=IngestResponse)
async def ingest_pdf(files: List[UploadFile] = File(...)):
    """
    Upload and ingest PDF files.

    Args:
        files: List of PDF files to ingest

    Returns:
        IngestResponse with ingestion status
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    TEMP_DIR.mkdir(exist_ok=True)

    try:
        logger.info(f"📦 Processing {len(files)} PDF files")

        documents = []
        for file in files:
            if file.content_type != "application/pdf":
                raise HTTPException(
                    status_code=400,
                    detail=f"{file.filename} is not a PDF file"
                )

            temp_path = TEMP_DIR / file.filename
            content = await file.read()

            with open(temp_path, "wb") as f:
                f.write(content)

            pdf_docs = await PDFProcessor.extract_from_pdf(str(temp_path))

            for doc in pdf_docs:
                doc.metadata["source_file"] = file.filename
                doc.metadata["uploaded_at"] = datetime.now().isoformat()

            documents.extend(pdf_docs)
            temp_path.unlink()

        chunks = await pipeline.ingest_documents(documents)

        return IngestResponse(
            ingested_chunks=chunks,
            source_count=len(files),
            status="success",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error ingesting PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/web", response_model=IngestResponse)
async def ingest_web(request: WebScrapingRequest):
    """
    Scrape and ingest web content.

    Args:
        request: WebScrapingRequest with URLs to scrape

    Returns:
        IngestResponse with ingestion status
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        urls = [str(url) for url in request.urls]
        logger.info(f"🌐 Scraping {len(urls)} URLs")

        documents = await WebScraper.scrape_multiple(
            urls,
            request.timeout,
            request.include_links
        )

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No content could be scraped from the provided URLs"
            )

        chunks = await pipeline.ingest_documents(documents)

        return IngestResponse(
            ingested_chunks=chunks,
            source_count=len(documents),
            status="success",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error scraping web: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))