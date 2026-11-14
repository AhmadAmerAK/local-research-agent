"""Custom Bounded OpenAlex/Tavily HTTP tools"""
import asyncio
import json
import logging
import time
from typing import Callable
from threading import Event, RLock

import httpx

from .config import Settings
from .evidence import EvidenceStore, normalise_openalex, normalise_tavily
from .cancellation import RequestCancelled, cancellable

LOGGER = logging.getLogger(__name__)
PROVIDER_TIMEOUT = httpx.Timeout(connect=5, read=10, write=10, pool=5)
PROVIDER_DEADLINE = 30.0
OPENALEX_URL = "https://api.openalex.org/works"
TAVILY_URL = "https://api.tavily.com/search"
# Supported API maxima, not application research budgets:
# https://help.openalex.org/api/paging/
# https://docs.tavily.com/documentation/api-reference/endpoint/search
PROVIDER_RESULT_LIMITS = {"openalex": 100, "tavily": 20}


class ResearchAPIError(RuntimeError):
    pass


def _request(client: httpx.Client, method: str, url: str, **kwargs) -> dict:
    """Retry transient failures without leaking credentials to the model or logs."""
    for attempt in range(3):
        try:
            response = client.request(method, url, timeout=PROVIDER_TIMEOUT, **kwargs)
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


def _openalex_request(query, key, count):
    params = {
        "search": query,
        "per_page": min(max(count, 1), PROVIDER_RESULT_LIMITS["openalex"]),
        "select": "id,doi,title,display_name,publication_year,authorships,abstract_inverted_index,cited_by_count",
    }
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return "GET", OPENALEX_URL, {"params": params, "headers": headers}


def _tavily_request(query, key, count):
    if not key:
        raise ResearchAPIError("TAVILY_API_KEY is missing (required for general web research)")
    payload = {
        "query": query,
        "topic": "general",
        "search_depth": "basic",
        "max_results": min(max(count, 1), PROVIDER_RESULT_LIMITS["tavily"]),
        "auto_parameters": False,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }
    return "POST", TAVILY_URL, {"json": payload, "headers": {
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"}}


def _results(data, provider):
    results = data.get("results", [])
    if not isinstance(results, list):
        raise ResearchAPIError(f"Malformed {provider} results")
    return [item for item in results if isinstance(item, dict)]


def _search_sync(request, provider, client):
    owned = client is None
    client = client or httpx.Client()
    try:
        method, url, kwargs = request
        return _results(_request(client, method, url, **kwargs), provider)
    finally:
        if owned:
            client.close()


def openalex_search(query: str, key: str, count: int = 5,
                    client: httpx.Client | None = None) -> list[dict]:
    """Compatibility API; native agent tools use the cancellable async version."""
    return _search_sync(_openalex_request(query, key, count), "OpenAlex", client)


def tavily_search(query: str, key: str, count: int = 5,
                  client: httpx.Client | None = None) -> list[dict]:
    return _search_sync(_tavily_request(query, key, count), "Tavily", client)


async def _request_async(client, method, url, **kwargs):
    for attempt in range(3):
        try:
            response = await client.request(method, url, timeout=PROVIDER_TIMEOUT, **kwargs)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                await asyncio.sleep(.5 * 2 ** attempt)
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
            await asyncio.sleep(.5 * 2 ** attempt)
        except ValueError as exc:
            raise ResearchAPIError("Research provider returned invalid JSON") from exc


async def _search_async(request, provider, client, cancel_event):
    owned = client is None
    client = client or httpx.AsyncClient()
    try:
        method, url, kwargs = request
        async with asyncio.timeout(PROVIDER_DEADLINE):
            data = await cancellable(_request_async(client, method, url, **kwargs), cancel_event)
        return _results(data, provider)
    except TimeoutError as exc:
        raise ResearchAPIError("Research service unreachable or timed out") from exc
    finally:
        if owned:
            await client.aclose()


async def openalex_search_async(query, key, count=5, client=None, cancel_event=None):
    return await _search_async(_openalex_request(query, key, count), "OpenAlex", client, cancel_event)


async def tavily_search_async(query, key, count=5, client=None, cancel_event=None):
    return await _search_async(_tavily_request(query, key, count), "Tavily", client, cancel_event)


class ResearchTools:
    """Per-run tool budget and local persistence: this is the Python policy boundary."""

    def __init__(self, settings: Settings, store: EvidenceStore,
                 on_status: Callable[[str], None] | None = None,
                 client: httpx.Client | None = None):
        self.settings, self.store, self.client = settings, store, client
        self.used_queries = 0
        self.used_this_round = 0
        self.queries: list[str] = []
        self.provider_errors: list[str] = []
        self.enforce_route = False
        self.active_route: str | None = None
        self.disclosed_ids: set[str] = set()
        self._budget_lock = RLock()
        self.cancel_event = Event()
        self.on_status = on_status or (lambda _: None)

    def start_round(self) -> None:
        self.used_this_round = 0

    def start_turn(self) -> None:
        """Renew the user-turn budget without discarding session evidence or IDs."""
        with self._budget_lock:
            self.used_queries = self.used_this_round = 0
            self.queries.clear()
            self.provider_errors.clear()
            self.active_route = None

    def _search(self, provider: str, query: str) -> str:
        with self._budget_lock:
            return self._search_permitted(provider, query)

    def _search_permitted(self, provider: str, query: str) -> str:
        prepared = self._prepare_search(provider, query)
        if isinstance(prepared, dict):
            return json.dumps(prepared)
        query = prepared
        try:
            count = self.settings.results_per_query
            search = openalex_search if provider == "openalex" else tavily_search
            key = self.settings.openalex_api_key if provider == "openalex" else self.settings.tavily_api_key
            return self._persist(provider, query, search(query, key, count, self.client))
        except ResearchAPIError as exc:
            return self._provider_error(provider, exc)

    def _prepare_search(self, provider, query):
        if self.cancel_event.is_set():
            return {"error": "Research cancelled", "provider": provider}
        if self.enforce_route and provider != {"academic": "openalex", "general": "tavily"}.get(self.active_route):
            return {"error": "Provider not permitted for this turn", "provider": provider}
        if self.provider_errors:
            return {"error": self.provider_errors[-1], "provider": provider}
        query = " ".join(query.split()).strip()
        if not query:
            return {"error": "Search query cannot be blank"}
        if len(query) > self.settings.max_query_chars:
            return {"error": f"Search query exceeds {self.settings.max_query_chars} characters; "
                             "shorten it or split it into focused queries."}
        if ((not self.enforce_route and self.used_this_round >= self.settings.max_search_queries) or
                self.used_queries >= self.settings.search_budget):
            return {"error": "Search budget exceeded"}
        self.used_queries += 1
        self.used_this_round += 1
        self.queries.append(query)
        self.on_status(f"Searching {provider}: {query}")
        return query

    async def search_async(self, provider, query, cancel_event=None):
        """Async native tool: no sync HTTP executor survives application shutdown."""
        signal = cancel_event if cancel_event is not None else self.cancel_event
        with self._budget_lock:
            prepared = self._prepare_search(provider, query)
        if isinstance(prepared, dict):
            return json.dumps(prepared)
        try:
            search = openalex_search_async if provider == "openalex" else tavily_search_async
            key = self.settings.openalex_api_key if provider == "openalex" else self.settings.tavily_api_key
            raw = await search(prepared, key, self.settings.results_per_query, cancel_event=signal)
            # A received response is committed without an await between raw data
            # and IDs. Cancellation cannot leave half a successful tool result.
            return self._persist(provider, prepared, raw)
        except RequestCancelled:
            return json.dumps({"error": "Research cancelled", "provider": provider})
        except ResearchAPIError as exc:
            return self._provider_error(provider, exc)

    def _persist(self, provider, query, raw):
        normalize = normalise_openalex if provider == "openalex" else normalise_tavily
        normalized = [normalize(x, snippet_chars=self.settings.search_snippet_chars) for x in raw]
        with self._budget_lock:
            self.store.save_raw(provider, query, raw)
            for item in normalized:
                if item is not None:
                    item.query = query
            added = self.store.add([x for x in normalized if x is not None])
            self.disclosed_ids.update(item.source_id for item in added)
        self.on_status(f"Collected {len(added)} new sources ({len(self.store.sources)} total)")
        return json.dumps({"provider": provider, "query": query,
                           "requested_results": self.settings.results_per_query,
                           "result_limit": min(self.settings.results_per_query, PROVIDER_RESULT_LIMITS[provider]),
                           "snippet_char_limit": self.settings.search_snippet_chars, "new_sources": [
            {"id": item.source_id, "title": item.title, "url": item.url,
             "snippet": item.snippet} for item in added
        ]}, ensure_ascii=False)

    def _provider_error(self, provider, exc):
        self.provider_errors.append(str(exc))
        LOGGER.warning("%s search failed: %s", provider, exc)
        self.on_status(str(exc))
        return json.dumps({"error": str(exc), "provider": provider})

    def search_papers(self, query: str) -> str:
        return self._search("openalex", query)

    def search_web(self, query: str) -> str:
        return self._search("tavily", query)

    def strand_tools(self, route: str) -> list:
        # Register exactly ONE provider so the model cannot override Python's routing.
        from strands import tool, ToolContext

        def framework_result(payload: str):
            if "error" in json.loads(payload):
                return {"status": "error", "content": [{"text": payload}]}
            return payload

        if route == "academic":
            @tool(context=True)
            async def search_papers(query: str, tool_context: ToolContext) -> str | dict:
                """Search OpenAlex scientific/academic publications by topic or question."""
                return framework_result(await self.search_async("openalex", query, tool_context.cancel_signal))
            return [search_papers]
        if route == "general":
            @tool(context=True)
            async def search_web(query: str, tool_context: ToolContext) -> str | dict:
                """Search the general web through Tavily for evidence-backed research."""
                return framework_result(await self.search_async("tavily", query, tool_context.cancel_signal))
            return [search_web]
        return []
