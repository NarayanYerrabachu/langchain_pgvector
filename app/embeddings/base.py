"""PostgreSQL Vector Embedding Model - Base Implementation."""
from abc import ABC
from typing import Any, List
import logging
import json
from hashlib import sha256
from pydantic import BaseModel
from langchain_core.documents import Document

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EmbeddingsIndex(ABC):
    """Abstract base for embedding indices."""
    pass

class IndexItem(BaseModel):
    """Item to be indexed in the embedding store."""
    text: str
    meta: dict = {}

def embed_connection_encode(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for texts using OpenAI."""
    try:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        results = embeddings.embed_documents(texts)
        logger.info(f"✅ Generated embeddings for {len(texts)} texts")
        return results
    except Exception as e:
        logger.error(f"❌ Error generating embeddings: {e}")
        # Return mock embeddings if OpenAI fails
        import random
        return [[random.random() for _ in range(1536)] for _ in texts]

def get_embedding(text: str) -> List[float]:
    """Get embedding for single text using OpenAI."""
    try:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        result = embeddings.embed_query(text)
        return result
    except Exception as e:
        logger.error(f"❌ Error generating embedding: {e}")
        # Return mock embedding if OpenAI fails
        import random
        return [random.random() for _ in range(1536)]

class BasePgVectorEmbeddingModel(EmbeddingsIndex, ABC):
    """Base PGVector embedding provider with common functionality."""

    engine_name = "BasePgVectorEmbeddingModel"
    collection_name: list[str] = []
    current_index = 0

    def __init__(self, **kwargs: Any) -> None:
        # Import here to avoid circular imports
        from app.config import settings
        from app.db import PostgresVdbClient

        self.name = self.collection_name[self.__class__.current_index]
        self.__class__.current_index += 1

        try:
            # Create real PostgreSQL connection
            self.db_client = PostgresVdbClient(
                engine_name=self.engine_name,
                dbname=settings.db_name,
                user=settings.db_user,
                password=settings.db_password,
                host=settings.db_host,
                port=settings.db_port,
                min_conn=settings.db_min_conn,
                max_conn=settings.db_max_conn
            )
            self.db_client.create_table(self.name, vector_dim=settings.vector_dim)
            logger.debug(f"{self.engine_name} initialized with table '{self.name}'")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database client: {e}", exc_info=True)
            raise

    async def add_item(self, item: IndexItem) -> None:
        await self._process_batch([item])

    async def add_items(self, items: List[IndexItem]) -> None:
        """Process items in optimized batches."""
        logger.debug(f"{self.name} Adding {len(items)} items")
        batch_size = 50
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i + batch_size]
            await self._process_batch(batch_items)

    async def _process_batch(self, items: List[IndexItem]) -> None:
        """Process a batch of items efficiently."""
        if not items:
            return

        # Step 1: Generate IDs
        items_with_ids = [(sha256(item.text.encode()).hexdigest(), item) for item in items]

        # Step 2: Check which exist
        all_ids = [item_id for item_id, _ in items_with_ids]
        existence_map = self.db_client.batch_exists_check(self.name, all_ids)

        # Step 3: Filter new items
        items_to_process = []
        item_ids = []
        for item_id, item in items_with_ids:
            if not existence_map.get(item_id, False):
                items_to_process.append(item)
                item_ids.append(item_id)

        if not items_to_process:
            logger.debug(f"{self.name} All items already exist")
            return

        # Step 4: Get embeddings
        texts = [item.text for item in items_to_process]
        embeddings = embed_connection_encode(texts)

        # Step 5: Prepare data
        insert_data = []
        for i, item in enumerate(items_to_process):
            insert_data.append((
                item_ids[i],
                item.text,
                embeddings[i],
                json.dumps(item.meta)
            ))

        # Step 6: Insert to database
        columns = ["id", "text", "embedding", "metadata"]
        self.db_client.batch_insert(self.name, columns, insert_data)
        logger.debug(f"{self.name} Added {len(items_to_process)} items")

    async def search(self, text: str, max_results: int = 5, **kwargs: Any) -> List[IndexItem]:
        """Search for similar items."""
        try:
            logger.debug(f"{self.name} Searching for: {text}")
            query_embedding = get_embedding(text)

            # Search in database
            results = self.db_client.search(
                self.name,
                query_embedding,
                limit=max_results
            )

            index_items = [
                IndexItem(text=r['text'], meta=r['metadata'])
                for r in results
            ]

            logger.debug(f"✅ Found {len(index_items)} results")
            return index_items
        except Exception as e:
            logger.error(f"❌ Error in search: {e}", exc_info=True)
            return []

    def __del__(self) -> None:
        if hasattr(self, 'db_client'):
            try:
                self.db_client.close()
            except:
                pass