"""PDF document processing module."""
from langchain_community.document_loaders import PyPDFLoader

import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handle PDF file extraction and processing."""

    @staticmethod
    async def extract_from_pdf(file_path: str) -> list[Document]:
        """
        Extract text content from PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            List of Document objects, one per page

        Raises:
            Exception: If PDF extraction fails
        """
        try:
            logger.info(f"📄 Extracting from PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"✅ Extracted {len(documents)} pages from {file_path}")
            return documents
        except Exception as e:
            logger.error(f"❌ Error extracting PDF: {e}", exc_info=True)
            raise