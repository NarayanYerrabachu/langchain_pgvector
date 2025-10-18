"""Vector retriever for LangChain."""
from typing import Any, List, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
import asyncio
import logging

logger = logging.getLogger(__name__)

class PgVectorRetriever(BaseRetriever):
    """Custom retriever using PostgreSQL pgvector."""

    # Declare fields for Pydantic model
    embedding_model: Any = None
    k: int = 4

    def __init__(self, embedding_model: Any, k: int = 4, **kwargs):
        """Initialize retriever with embedding model."""
        # Initialize parent Pydantic model properly
        super().__init__(embedding_model=embedding_model, k=k, **kwargs)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async retrieval of relevant documents."""
        logger.debug(f"🔍 Searching for: {query}")
        try:
            results = await self.embedding_model.search(query, max_results=self.k)
            docs = [
                Document(
                    page_content=item.text,
                    metadata=item.meta if isinstance(item.meta, dict) else {}
                )
                for item in results
            ]
            logger.debug(f"✅ Found {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"❌ Error in async search: {e}", exc_info=True)
            return []

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Synchronous wrapper for async search."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            results = loop.run_until_complete(
                self.embedding_model.search(query, max_results=self.k)
            )
            docs = [
                Document(
                    page_content=item.text,
                    metadata=item.meta if isinstance(item.meta, dict) else {}
                )
                for item in results
            ]
            return docs
        except Exception as e:
            logger.error(f"❌ Error in sync search: {e}", exc_info=True)
            return []
