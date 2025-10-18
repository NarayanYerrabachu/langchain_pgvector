"""Application Entry Point."""
import uvicorn
import logging

# Import config first
from app.config import settings
from app.utils.logger import get_logger

# Then import other modules
from app import create_app
from app.rag.pipeline import RagPipeline
from app.embeddings import BasePgVectorEmbeddingModel
from app.routes import ingest, query, health

logger = get_logger(__name__)

pipeline = None


async def init_pipeline():
    """Initialize the RAG pipeline on startup."""
    global pipeline

    try:
        logger.info("🔧 Initializing RAG pipeline...")

        class DocumentEmbeddingModel(BasePgVectorEmbeddingModel):
            engine_name = "rag_embeddings"
            collection_name = ["documents"]

        embedding_model = DocumentEmbeddingModel()
        logger.info("✅ Embedding model initialized")

        pipeline = RagPipeline(
            embedding_model=embedding_model,
            llm_model=settings.llm_model,
            temperature=settings.llm_temperature
        )
        logger.info("✅ RAG pipeline initialized")

        ingest.pipeline = pipeline
        query.pipeline = pipeline
        health.pipeline = pipeline
        logger.info("✅ Pipeline injected into routes")

    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}", exc_info=True)
        raise


def create_application():
    """Create and configure FastAPI application."""
    app = create_app()

    @app.on_event("startup")
    async def startup():
        await init_pipeline()
        logger.info("🚀 Application started")

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("🛑 Shutting down application")

    return app


# ✅ Create app at module level
app = create_application()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=settings.fastapi_reload,
        log_level=settings.log_level.lower()
    )