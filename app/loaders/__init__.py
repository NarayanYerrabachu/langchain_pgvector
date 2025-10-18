"""Loaders package."""
from app.loaders.pdf import PDFProcessor
from app.loaders.web import WebScraper

__all__ = ["PDFProcessor", "WebScraper"]