"""Search engine interface for research agents.

Domain researchers register engines for their domain (arxiv, familysearch,
etc.). The framework routes search queries to registered engines and
merges results.

Two default engines ship with researcher-lib:
- ``WebSearchEngine`` — DuckDuckGo instant answers (no API key)
- ``WebFetchEngine`` — fetch any URL, extract text

Domain-specific engines live in their own packages, not here.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single result from a search engine."""

    title: str
    url: str
    snippet: str = ""
    source: str = ""  # engine name that produced this
    score: float = 0.0  # relevance score if the engine provides one
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSearchEngine(ABC):
    """Interface for pluggable search engines.

    Subclass and implement ``search()``. Register with an
    ``EngineRegistry`` so ``BaseResearchAgent`` can route queries.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    async def search(
        self, query: str, max_results: int = 10, **kwargs: Any
    ) -> list[SearchResult]:
        """Search for documents matching the query."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


class EngineRegistry:
    """Config-driven registry of search engines.

    Engines register by name. Callers search across all registered
    engines (or a specific subset) and get merged results.
    """

    def __init__(self) -> None:
        self._engines: dict[str, BaseSearchEngine] = {}

    def register(self, engine: BaseSearchEngine) -> None:
        """Register an engine by its name."""
        if not engine.name:
            raise ValueError(f"Engine {engine!r} has no name")
        self._engines[engine.name] = engine
        logger.debug("Registered search engine: %s", engine.name)

    def get(self, name: str) -> Optional[BaseSearchEngine]:
        """Get a registered engine by name, or None."""
        return self._engines.get(name)

    def list_engines(self) -> list[str]:
        """Return names of all registered engines."""
        return list(self._engines.keys())

    def has(self, name: str) -> bool:
        return name in self._engines

    async def search(
        self,
        query: str,
        engines: Optional[list[str]] = None,
        max_results: int = 20,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Search across registered engines, merge results.

        Args:
            query: Search query.
            engines: Engine names to use (default: all registered).
            max_results: Max results per engine.

        Returns:
            Merged results from all queried engines, tagged with source.
        """
        import asyncio

        targets = engines or list(self._engines.keys())
        tasks = []
        for name in targets:
            engine = self._engines.get(name)
            if engine is None:
                logger.warning("Engine %r not registered, skipping", name)
                continue
            tasks.append(self._search_one(engine, query, max_results, **kwargs))

        if not tasks:
            return []

        results_per_engine = await asyncio.gather(*tasks, return_exceptions=True)
        merged: list[SearchResult] = []
        for result in results_per_engine:
            if isinstance(result, Exception):
                logger.warning("Engine search failed: %s", result)
                continue
            merged.extend(result)

        return merged

    @staticmethod
    async def _search_one(
        engine: BaseSearchEngine,
        query: str,
        max_results: int,
        **kwargs: Any,
    ) -> list[SearchResult]:
        results = await engine.search(query, max_results=max_results, **kwargs)
        for r in results:
            if not r.source:
                r.source = engine.name
        return results


# ---------------------------------------------------------------------------
# Default engines (ship with researcher-lib, no API keys needed)
# ---------------------------------------------------------------------------


class WebSearchEngine(BaseSearchEngine):
    """DuckDuckGo instant answers — lightweight, no API key.

    Uses the DuckDuckGo HTML endpoint for basic web search results.
    Not as comprehensive as Google but free and keyless.
    """

    name = "web_search"
    description = "Web search via DuckDuckGo instant answers"

    async def search(
        self, query: str, max_results: int = 10, **kwargs: Any
    ) -> list[SearchResult]:
        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not installed — web_search unavailable")
            return []

        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, data={"q": query}, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status != 200:
                        return []
                    html = await resp.text()
        except Exception as e:
            logger.warning("web_search failed for %r: %s", query, e)
            return []

        return self._parse_ddg_html(html, max_results)

    @staticmethod
    def _parse_ddg_html(html: str, max_results: int) -> list[SearchResult]:
        """Extract results from DuckDuckGo HTML response."""
        import re

        results: list[SearchResult] = []
        # DuckDuckGo HTML results are in <a class="result__a" href="...">
        pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
            r'.*?<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        for match in pattern.finditer(html):
            if len(results) >= max_results:
                break
            url = match.group(1)
            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            snippet = re.sub(r"<[^>]+>", "", match.group(3)).strip()
            if url and title:
                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="web_search",
                    )
                )
        return results


class WebFetchEngine(BaseSearchEngine):
    """Fetch any URL and extract text content.

    Not a search engine per se — it fetches a specific URL. Useful for
    on-demand ingestion of a known document.
    """

    name = "web_fetch"
    description = "Fetch a URL and extract text content"

    async def search(
        self, query: str, max_results: int = 1, **kwargs: Any
    ) -> list[SearchResult]:
        """``query`` is treated as a URL to fetch."""
        try:
            import aiohttp
        except ImportError:
            return []

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    query, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 200:
                        return []
                    text = await resp.text()
        except Exception as e:
            logger.warning("web_fetch failed for %r: %s", query, e)
            return []

        # Strip HTML tags for a rough text extraction
        import re

        clean = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        clean = re.sub(r"<style[^>]*>.*?</style>", "", clean, flags=re.DOTALL)
        clean = re.sub(r"<[^>]+>", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        title_match = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else query

        return [
            SearchResult(
                title=title,
                url=query,
                snippet=clean[:500],
                source="web_fetch",
                metadata={"full_text_length": len(clean)},
            )
        ]
