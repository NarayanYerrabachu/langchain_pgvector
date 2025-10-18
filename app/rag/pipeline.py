"""RAG pipeline orchestration."""
from typing import List
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.retriever import PgVectorRetriever
from app.embeddings import IndexItem
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RagPipeline:
    """Complete RAG pipeline combining embeddings, retrieval, and LLM."""

    def __init__(
        self,
        embedding_model,
        llm_model: str = "gpt-4-turbo",
        temperature: float = 0.7
    ):
        """Initialize RAG pipeline."""
        self.embedding_model = embedding_model
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        # Create retriever with proper initialization
        self.retriever = PgVectorRetriever(
            embedding_model=embedding_model,
            k=4
        )
        logger.info("✅ Retriever created")

        # Simple query pipeline without RetrievalQA (to avoid compatibility issues)
        self.qa_chain = None
        logger.info("✅ RAG Pipeline initialized")

    async def ingest_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000
    ) -> int:
        """Ingest documents into vector store."""
        logger.info(f"📥 Ingesting {len(documents)} documents...")

        self.text_splitter.chunk_size = chunk_size
        split_docs = self.text_splitter.split_documents(documents)

        items = [
            IndexItem(
                text=doc.page_content,
                meta={
                    **doc.metadata,
                    "ingested_at": datetime.now().isoformat()
                }
            )
            for doc in split_docs
        ]

        await self.embedding_model.add_items(items)
        logger.info(f"✅ Ingested {len(items)} chunks")
        return len(items)

    async def query(self, question: str, top_k: int = 4) -> dict:
        """Query the RAG system."""
        logger.info(f"❓ Querying: {question}")

        try:
            # Get relevant documents
            retrieved_docs = await self.retriever._aget_relevant_documents(question)

            # Create context from documents
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Create prompt
            prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""

            # Get answer from LLM
            result = await asyncio.to_thread(
                lambda: self.llm.invoke(prompt)
            )

            response = {
                "question": question,
                "answer": result.content if hasattr(result, 'content') else str(result),
                "sources": [
                    {
                        "content": doc.page_content[:300],
                        "metadata": doc.metadata
                    }
                    for doc in retrieved_docs
                ],
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"✅ Query completed with {len(retrieved_docs)} sources")
            return response
        except Exception as e:
            logger.error(f"❌ Error querying: {e}", exc_info=True)
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }