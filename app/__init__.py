"""FastAPI application factory."""
from fastapi import FastAPI

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    # Import routes here to avoid circular imports
    from app.routes import health, ingest, query

    app = FastAPI(
        title="LangChain RAG API",
        description="Retrieval-Augmented Generation API with PDF & Web Scraping",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    @app.get("/")
    async def root():
        return {
            "message": "LangChain RAG API",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/health",
            "endpoints": {
                "GET /health": "Health check",
                "POST /ingest/texts": "Ingest plain text",
                "POST /ingest/pdf": "Upload and ingest PDF files",
                "POST /ingest/web": "Scrape and ingest web content",
                "POST /query": "Query the RAG system"
            }
        }

    app.include_router(health.router)
    app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
    app.include_router(query.router, tags=["Query"])

    return app