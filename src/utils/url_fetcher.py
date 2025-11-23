"""Utility for fetching content from URLs for RAG metrics calculation."""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
from loguru import logger


def _extract_text_from_html(html_content: str) -> str:
    """
    Extract text content from HTML.

    Args:
        html_content: Raw HTML content

    Returns:
        Extracted text content
    """
    # Remove script and style tags and their content
    # Use \s* to handle whitespace in closing tags (e.g., </script >)
    content = re.sub(
        r"<script[^>]*>.*?</script\s*>", "", html_content, flags=re.DOTALL | re.IGNORECASE
    )
    content = re.sub(
        r"<style[^>]*>.*?</style\s*>", "", content, flags=re.DOTALL | re.IGNORECASE
    )

    # Remove HTML tags
    content = re.sub(r"<[^>]+>", " ", content)

    # Decode HTML entities (basic ones)
    html_entities = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&nbsp;": " ",
        "&apos;": "'",
        "&mdash;": "—",
        "&ndash;": "–",
    }
    for entity, char in html_entities.items():
        content = content.replace(entity, char)

    # Clean up whitespace
    content = re.sub(r"\s+", " ", content)
    return content.strip()


def _extract_text_from_pdf(pdf_content: bytes) -> str | None:
    """
    Extract text from PDF content (basic approach without external libraries).

    Args:
        pdf_content: Raw PDF bytes

    Returns:
        Extracted text or None if extraction fails
    """
    try:
        # Try to extract text from PDF using basic regex (works for text-based PDFs)
        # This is a simple approach - for better results, consider using PyPDF2 or pdfplumber
        text = pdf_content.decode("utf-8", errors="ignore")

        # Look for text streams in PDF
        # PDF text is often in streams like: (text content) Tj or [text] TJ
        text_patterns = [
            r"\((.*?)\)\s*Tj",  # Simple text: (text) Tj
            r"\[(.*?)\]\s*TJ",  # Array text: [text] TJ
        ]

        extracted_text = []
        for pattern in text_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the extracted text
                cleaned = re.sub(r"\\[a-zA-Z0-9]+", " ", match)  # Remove escape sequences
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                if cleaned and len(cleaned) > 3:  # Filter out very short fragments
                    extracted_text.append(cleaned)

        if extracted_text:
            return " ".join(extracted_text[:1000])  # Limit to avoid huge strings
        return None
    except Exception as e:
        logger.debug(f"Failed to extract text from PDF: {e}")
        return None


def fetch_url_content(
    url: str,
    timeout: int = 5,
    max_length: int = 10000,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> str | None:
    """
    Fetch text content from a URL with retry logic and comprehensive error handling.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds (default: 5)
        max_length: Maximum content length to return (default: 10000 chars)
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 1.0)

    Returns:
        Extracted text content, or None if fetch failed after all retries
    """
    if not url or not isinstance(url, str) or not url.strip():
        return None

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        logger.debug(f"Skipping non-HTTP URL: {url}")
        return None

    # Normalize URL (handle encoding issues)
    try:
        parsed = urlparse(url)
        url = parsed.geturl()
    except Exception:
        pass  # Use URL as-is if parsing fails

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; PyAI-Slayer/1.0; +https://github.com/pyai-slayer)",
        "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf,text/plain,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                # Exponential backoff for retries
                delay = retry_delay * (2 ** (attempt - 1))
                logger.debug(
                    f"Retrying URL {url} (attempt {attempt + 1}/{max_retries}) after {delay}s"
                )
                time.sleep(delay)

            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
                stream=False,  # Load entire response
            )
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("Content-Type", "").lower()

            # Handle different content types
            if "text/html" in content_type or "application/xhtml" in content_type:
                # HTML content
                content = _extract_text_from_html(response.text)
            elif "text/plain" in content_type:
                # Plain text
                content = response.text.strip()
            elif "application/pdf" in content_type or url.lower().endswith(".pdf"):
                # PDF content
                logger.debug(f"Attempting to extract text from PDF: {url}")
                pdf_text = _extract_text_from_pdf(response.content)
                if pdf_text:
                    content = pdf_text
                else:
                    # Fallback: try to extract any readable text from PDF
                    try:
                        # Some PDFs have readable text when decoded
                        text = response.content.decode("utf-8", errors="ignore")
                        # Extract text between common PDF text markers
                        text_matches = re.findall(r"\((.*?)\)", text)
                        if text_matches:
                            content = " ".join([m for m in text_matches if len(m) > 3])[:max_length]
                        else:
                            logger.debug(f"Could not extract text from PDF: {url}")
                            return None
                    except Exception:
                        logger.debug(f"Failed to decode PDF content: {url}")
                        return None
            elif "application/json" in content_type:
                # JSON content - try to extract text fields
                try:
                    import json

                    data = response.json()
                    # Try to find text content in JSON
                    if isinstance(data, dict):
                        # Look for common text fields
                        text_fields = ["text", "content", "body", "description", "summary", "title"]
                        content_parts = []
                        for field in text_fields:
                            if field in data and isinstance(data[field], str):
                                content_parts.append(data[field])
                        if content_parts:
                            content = " ".join(content_parts)
                        else:
                            # Fallback: stringify the JSON
                            content = json.dumps(data, ensure_ascii=False)
                    elif isinstance(data, str):
                        content = data
                    else:
                        content = str(data)
                except Exception:
                    content = response.text
            else:
                # Try to extract text anyway (might be HTML without proper content-type)
                logger.debug(
                    f"Unknown content type '{content_type}' for {url}, attempting text extraction"
                )
                try:
                    content = _extract_text_from_html(response.text)
                except Exception:
                    # Last resort: try as plain text
                    content = response.text.strip()

            if not content or not content.strip():
                logger.debug(f"Empty content extracted from {url}")
                if attempt < max_retries - 1:
                    continue  # Retry
                return None

            # Clean up and limit length
            content = re.sub(r"\s+", " ", content)
            content = content.strip()

            if len(content) > max_length:
                content = content[:max_length] + "..."

            logger.debug(f"Successfully fetched {len(content)} chars from {url}")
            return content

        except requests.exceptions.Timeout:
            last_error = "timeout"
            logger.debug(f"Timeout fetching URL (attempt {attempt + 1}/{max_retries}): {url}")
            if attempt < max_retries - 1:
                continue
        except requests.exceptions.ConnectionError as e:
            last_error = f"connection error: {e}"
            logger.debug(
                f"Connection error fetching URL (attempt {attempt + 1}/{max_retries}): {url} - {e}"
            )
            if attempt < max_retries - 1:
                continue
        except requests.exceptions.HTTPError as e:
            # Don't retry on client errors (4xx), but retry on server errors (5xx)
            if e.response and e.response.status_code >= 500:
                last_error = f"HTTP {e.response.status_code}"
                logger.debug(
                    f"Server error fetching URL (attempt {attempt + 1}/{max_retries}): {url} - {e}"
                )
                if attempt < max_retries - 1:
                    continue
            else:
                logger.debug(f"HTTP error fetching URL: {url} - {e}")
                return None
        except requests.exceptions.RequestException as e:
            last_error = f"request error: {e}"
            logger.debug(
                f"Request error fetching URL (attempt {attempt + 1}/{max_retries}): {url} - {e}"
            )
            if attempt < max_retries - 1:
                continue
        except Exception as e:
            last_error = f"unexpected error: {e}"
            logger.debug(
                f"Unexpected error fetching URL (attempt {attempt + 1}/{max_retries}): {url} - {e}"
            )
            if attempt < max_retries - 1:
                continue

    # All retries exhausted
    logger.debug(
        f"Failed to fetch URL after {max_retries} attempts: {url} (last error: {last_error})"
    )
    return None


def fetch_urls_content(
    urls: list[str],
    timeout: int = 5,
    max_length: int = 10000,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    max_workers: int = 5,
) -> list[str]:
    """
    Fetch content from multiple URLs in parallel with comprehensive error handling.

    Args:
        urls: List of URLs to fetch
        timeout: Request timeout per URL in seconds (default: 5)
        max_length: Maximum content length per URL (default: 10000 chars)
        max_retries: Maximum number of retry attempts per URL (default: 3)
        retry_delay: Delay between retries in seconds (default: 1.0)
        max_workers: Maximum number of parallel workers (default: 5)

    Returns:
        List of extracted text content (None values filtered out)
    """
    if not urls:
        return []

    results = []
    # Use ThreadPoolExecutor for parallel fetching
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all fetch tasks
        future_to_url = {
            executor.submit(
                fetch_url_content,
                url,
                timeout,
                max_length,
                max_retries,
                retry_delay,
            ): url
            for url in urls
        }

        # Collect results as they complete
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                content = future.result()
                if content:
                    results.append(content)
            except Exception as e:
                logger.debug(f"Exception fetching URL {url}: {e}")

    return results
