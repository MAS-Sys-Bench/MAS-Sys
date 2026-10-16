"""Utility: Fetch academic papers from the arXiv public API.

Calls the public arXiv Atom/API feed (no API key required) and returns
structured ResearchPaper objects.  This is a pure utility used internally
by PaperDB for RAG — it is NOT a BaseTool and cannot be called by agents.

Entry-point:
  ArxivSearch.fetch() — fetch papers from arxiv.org → List[ResearchPaper]

API reference: https://info.arxiv.org/help/api/index.html
"""

from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Optional

import httpx

from app.logger import logger
from app.schema_research import ResearchPaper

# arXiv Atom feed namespace
_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"

_BASE_URL = "https://export.arxiv.org/api/query"
_DEFAULT_TIMEOUT = 20  # seconds


class ArxivSearch:
    """Fetch paper metadata from arxiv.org (title, authors, abstract, URL).

    Uses the public arXiv Atom API — no API key required.
    Results can be injected into an LLM prompt or stored in PaperDB.
    """

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    async def fetch(
        self,
        query: str,
        num: int = 10,
        domain: Optional[str] = None,
        author: Optional[str] = None,
    ) -> List[ResearchPaper]:
        """Fetch papers from arXiv and return ResearchPaper objects.

        Args:
            query: Full-text search query.
            num: Max number of results (1–50).
            domain: arXiv category, e.g. ``"cs.AI"``.
            author: Filter by author name (added as ``au:`` clause).

        Returns:
            List of ResearchPaper objects, sorted by relevance (arXiv default).
        """
        num = min(max(1, num), 50)
        search_query = _build_query(query, domain=domain, author=author)
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": num,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"{_BASE_URL}?{urllib.parse.urlencode(params)}"
        logger.info(f"[ArxivSearch] GET {url}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                xml_text = resp.text
        except httpx.TimeoutException:
            logger.error("[ArxivSearch] request timed out.")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"[ArxivSearch] HTTP error: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"[ArxivSearch] unexpected error: {e}")
            return []

        papers = _parse_atom(xml_text)
        logger.info(f"[ArxivSearch] fetched {len(papers)} papers.")
        return papers


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_query(
    query: str,
    domain: Optional[str] = None,
    author: Optional[str] = None,
) -> str:
    """Compose an arXiv API search_query string.

    Examples::
        _build_query("attention transformer")
        → "all:attention+transformer"

        _build_query("attention transformer", domain="cs.AI", author="Vaswani")
        → "all:attention+transformer+AND+cat:cs.AI+AND+au:Vaswani"
    """
    parts: List[str] = [f"all:{urllib.parse.quote_plus(query)}"]
    if domain:
        parts.append(f"cat:{domain}")
    if author:
        parts.append(f"au:{urllib.parse.quote_plus(author)}")
    return "+AND+".join(parts)


def _parse_atom(xml_text: str) -> List[ResearchPaper]:
    """Parse arXiv Atom XML and return a list of ResearchPaper objects."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.error(f"[ArxivSearch] XML parse error: {e}")
        return []

    papers: List[ResearchPaper] = []
    for entry in root.findall(f"{{{_NS}}}entry"):
        title_el = entry.find(f"{{{_NS}}}title")
        abstract_el = entry.find(f"{{{_NS}}}summary")
        id_el = entry.find(f"{{{_NS}}}id")

        title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
        abstract = (abstract_el.text or "").strip().replace("\n", " ") if abstract_el is not None else ""
        arxiv_url = (id_el.text or "").strip() if id_el is not None else ""

        # Extract arXiv ID from URL like http://arxiv.org/abs/2305.12345v1
        arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else ""

        # Authors
        authors: List[str] = []
        for author_el in entry.findall(f"{{{_NS}}}author"):
            name_el = author_el.find(f"{{{_NS}}}name")
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        # Primary category tag (arxiv category)
        cat_el = entry.find(f"{{{_ARXIV_NS}}}primary_category")
        domain_str: Optional[str] = None
        if cat_el is not None:
            domain_str = cat_el.attrib.get("term")

        if not title or not abstract:
            continue

        papers.append(
            ResearchPaper(
                title=title,
                abstract=abstract,
                authors=authors,
                url=arxiv_url,
                arxiv_id=arxiv_id,
                domain=domain_str,
            )
        )

    return papers
