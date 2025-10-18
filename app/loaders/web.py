"""Web scraping module."""

from bs4 import BeautifulSoup
import requests
import asyncio
from datetime import datetime
import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class WebScraper:
    """Handle web scraping with BeautifulSoup."""

    @staticmethod
    async def scrape_url(
        url: str,
        timeout: int = 10,
        include_links: bool = False
    ) -> Document:
        """
        Scrape content from a single URL.

        Args:
            url: URL to scrape
            timeout: Request timeout in seconds
            include_links: Whether to include links in metadata

        Returns:
            Document object with scraped content

        Raises:
            Exception: If scraping fails
        """
        try:
            logger.info(f"🌐 Scraping: {url}")

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, "html.parser")

            # Remove unwanted elements
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()

            # Extract text
            text = soup.get_text(separator="\n", strip=True)

            # Extract title
            title = soup.title.string if soup.title else "Unknown"

            # Extract links if requested
            links = []
            if include_links:
                links = [a.get("href") for a in soup.find_all("a", href=True)]

            metadata = {
                "source": url,
                "title": title,
                "scraped_at": datetime.now().isoformat(),
                "links_count": len(links)
            }

            logger.info(f"✅ Scraped {len(text)} characters from {url}")
            return Document(page_content=text, metadata=metadata)

        except Exception as e:
            logger.error(f"❌ Error scraping {url}: {e}", exc_info=True)
            raise

    @staticmethod
    async def scrape_multiple(
        urls: list[str],
        timeout: int = 10,
        include_links: bool = False
    ) -> list[Document]:
        """
        Scrape multiple URLs concurrently.

        Args:
            urls: List of URLs to scrape
            timeout: Request timeout in seconds
            include_links: Whether to include links

        Returns:
            List of Document objects
        """
        logger.info(f"🌐 Scraping {len(urls)} URLs...")

        tasks = [
            WebScraper.scrape_url(url, timeout, include_links)
            for url in urls
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        documents = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"⚠️ Failed to scrape {urls[i]}: {result}")
            else:
                documents.append(result)

        logger.info(f"✅ Successfully scraped {len(documents)}/{len(urls)} URLs")
        return documents