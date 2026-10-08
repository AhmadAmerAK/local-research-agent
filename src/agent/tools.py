"""Custom Bounded OpenAlex/Tavily HTTP tools"""
import json
import logging
import time
from typing import Callable

import httpx

from .config import Settings
from .evidence import EvidenceStore, normalise_openalex, normalise_tavily

LOGGER = logging.getLogger(__name__)


class ResearchAPIError(RuntimeError):
    pass


def _request(client: httpx.Client, method: str, url: str, **kwargs) -> dict:
    """Retry transient failures without leaking credentials to the model or logs."""
    for attempt in range(3):
        try:
            response = client.request(method, url, timeout=25.0, **kwargs)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
                continue
            if response.status_code >= 400:
                raise ResearchAPIError(f"Research provider returned HTTP {response.status_code}")
            data = response.json()
            if not isinstance(data, dict):
                raise ResearchAPIError("Unexpected research provider response")
            return data
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == 2:
                raise ResearchAPIError("Research service unreachable or timed out") from exc
            time.sleep(0.5 * (2 ** attempt))
        except ValueError as exc:
            raise ResearchAPIError("Research provider returned invalid JSON") from exc
    raise ResearchAPIError("Research provider unavailable")


def openalex_search(query: str, key: str, count: int = 5,
                    client: httpx.Client | None = None) -> list[dict]:
    """GET /works?search=...; decode abstracts separately in evidence.py."""
    params = {
        "search": query,
        "per_page": min(max(count, 1), 25),
        "select": "id,doi,title,display_name,publication_year,authorships,abstract_inverted_index,cited_by_count",
    }
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    owned = client is None
    client = client or httpx.Client()
    try:
        result = _request(client, "GET", "https://api.openalex.org/works",
                          params=params, headers=headers)
        works = result.get("results", [])
        if not isinstance(works, list):
            raise ResearchAPIError("Malformed OpenAlex results")
        return [w for w in works if isinstance(w, dict)]
    finally:
        if owned:
            client.close()


def tavily_search(query: str, key: str, count: int = 5,
                  client: httpx.Client | None = None) -> list[dict]:
    """POST /search using the basic, lower-credit Tavily search."""
    if not key:
        raise ResearchAPIError("TAVILY_API_KEY is missing (required for general web research)")
    payload = {
        "query": query,
        "topic": "general",
        "search_depth": "basic",
        "max_results": min(max(count, 1), 20),
        "auto_parameters": False,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }
    owned = client is None
    client = client or httpx.Client()
    try:
        result = _request(client, "POST", "https://api.tavily.com/search",
                          json=payload,
                          headers={"Authorization": f"Bearer {key}",
                                   "Content-Type": "application/json"})
        hits = result.get("results", [])
        if not isinstance(hits, list):
            raise ResearchAPIError("Malformed Tavily results")
        return [r for r in hits if isinstance(r, dict)]
    finally:
        if owned:
            client.close()


class ResearchTools:
    """Per-run tool budget and local persistence: this is the Python policy boundary."""

    def __init__(self, settings: Settings, store: EvidenceStore,
                 on_status: Callable[[str], None] | None = None,
                 client: httpx.Client | None = None):
        self.settings, self.store, self.client = settings, store, client
        self.used_queries = 0
        self.used_this_round = 0
        self.queries: list[str] = []
        self.on_status = on_status or (lambda _: None)

    def start_round(self) -> None:
        self.used_this_round = 0

    def _search(self, provider: str, query: str) -> str:
        query = " ".join(query.split()).strip()[:300]
        if not query:
            return json.dumps({"error": "Search query cannot be blank"})
        if (self.used_this_round >= self.settings.max_search_queries or
                self.used_queries >= self.settings.max_search_queries * self.settings.max_research_rounds):
            return json.dumps({"error": "Search budget exceeded"})
        self.used_queries += 1
        self.used_this_round += 1
        self.queries.append(query)
        self.on_status(f"Searching {provider}: {query}")
        try:
            # Fetch several records, present a small set per call to the model.
            count = self.settings.max_sources
            if provider == "openalex":
                raw = openalex_search(query, self.settings.openalex_api_key, count, self.client)
                normalized = [normalise_openalex(x) for x in raw]
            else:
                raw = tavily_search(query, self.settings.tavily_api_key, count, self.client)
                normalized = [normalise_tavily(x) for x in raw]
            self.store.save_raw(provider, query, raw)
            for item in normalized:
                if item is not None:
                    item.query = query
            added = self.store.add([x for x in normalized if x is not None])
            self.on_status(f"Collected {len(added)} new sources ({len(self.store.sources)} total)")
            return json.dumps({"provider": provider, "query": query, "new_sources": [
                {"id": item.source_id, "title": item.title, "url": item.url,
                 "snippet": item.snippet[:700]} for item in added
            ]}, ensure_ascii=False)
        except ResearchAPIError as exc:
            LOGGER.warning("%s search failed: %s", provider, exc)
            self.on_status(str(exc))
            return json.dumps({"error": str(exc), "provider": provider})

    def search_papers(self, query: str) -> str:
        return self._search("openalex", query)

    def search_web(self, query: str) -> str:
        return self._search("tavily", query)

    def strand_tools(self, route: str) -> list:
        # Register exactly ONE provider so the model cannot override Python's routing.
        from strands import tool
        if route == "academic":
            @tool
            def search_papers(query: str) -> str:
                """Search OpenAlex scientific/academic publications by topic or question."""
                return self.search_papers(query)
            return [search_papers]
        if route == "general":
            @tool
            def search_web(query: str) -> str:
                """Search the general web through Tavily for evidence-backed research."""
                return self.search_web(query)
            return [search_web]
        return []
