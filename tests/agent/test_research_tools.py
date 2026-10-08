"""Offline provider-contract tests; no credits, no Ollama, no Strands installed needed."""
from dataclasses import replace
from pathlib import Path
import sys

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agent.config import Settings
from agent.evidence import (EvidenceStore, Source, abstract_from_inverted_index,
                            normalise_openalex, normalise_tavily)
from agent.tools import ResearchAPIError, ResearchTools, openalex_search, tavily_search
from agent.orchestrator import ResearchWorkflow, _extract_json


@pytest.fixture
def settings(monkeypatch):
    for key in ("MAX_SEARCH_QUERIES", "MAX_SOURCES", "MAX_RESEARCH_ROUNDS",
                "MIN_REQUIRED_SOURCES", "CONTEXT_WINDOW"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    return Settings.from_env()


def test_openalex_request_shape_and_headers():
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/works"
        assert request.url.params["search"] == "gait recognition"
        assert request.url.params["per_page"] == "3"
        assert request.headers["Authorization"] == "Bearer demo"
        assert "abstract_inverted_index" in request.url.params["select"]
        return httpx.Response(200, json={"results": [{"id": "https://openalex.org/W1", "title": "Paper"}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = openalex_search("gait recognition", "demo", 3, client)
    assert result[0]["title"] == "Paper"


def test_openalex_no_key_is_allowed():
    def handler(request):
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"results": []})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert openalex_search("research", "", client=client) == []


def test_tavily_contract():
    def handler(request):
        assert request.method == "POST" and request.url.path == "/search"
        assert request.headers["authorization"] == "Bearer demo"
        import json
        body = json.loads(request.content)
        assert body["search_depth"] == "basic"
        assert body["include_raw_content"] is False
        assert body["include_answer"] is False
        assert body["max_results"] == 5
        return httpx.Response(200, json={"results": [{
            "title": "Report", "url": "https://example.org/report", "content": "Useful", "score": 0.9}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = tavily_search("green hydrogen", "demo", client=client)
    assert len(result) == 1


def test_tavily_requires_key():
    with pytest.raises(ResearchAPIError, match="TAVILY_API_KEY"):
        tavily_search("hello", "")


def test_api_auth_failures_are_safe():
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(401, text="secret"))) as c:
        with pytest.raises(ResearchAPIError, match="401") as err:
            tavily_search("query", "secret", client=c)
    assert "secret" not in str(err.value)


def test_openalex_inverted_index_and_sources():
    data = {"id": "https://openalex.org/W1", "title": "A Test Paper", "publication_year": 2025,
            "authorships": [{"author": {"display_name": "Researcher"}}],
            "abstract_inverted_index": {"Gait": [0], "recognition": [1], "works": [2]}}
    assert abstract_from_inverted_index(data["abstract_inverted_index"]) == "Gait recognition works"
    normalized = normalise_openalex(data)
    assert normalized.snippet == "Gait recognition works"
    assert normalized.authors == ["Researcher"]
    assert normalized.year == 2025


def test_tavily_normalization():
    source = normalise_tavily({"title": "Public report", "url": "https://example.com/x",
                               "content": "Web finding", "published_date": "2026-10-01", "score": 0.95})
    assert source.provider == "tavily" and source.score == 0.95


def test_store_deduplicates_and_checks_ids(tmp_path):
    store = EvidenceStore(tmp_path)
    a = Source("", "tavily", "A", "https://a.example/path?ref=xyz", "test", [])
    b = Source("", "tavily", "A again", "https://a.example/path", "test2", [])
    assert len(store.add([a, b])) == 1
    cited, invalid = store.validate_citations("supported [S1], unsupported [S9]")
    assert cited == ["[S1]", "[S9]"] and invalid == ["[S9]"]
    assert (tmp_path / "sources.json").is_file()


def test_python_search_budget_saves_provider_results(tmp_path, settings):
    def handler(req):
        return httpx.Response(200, json={"results": [{"title": "Study", "id": "https://openalex.org/W1",
                                "abstract_inverted_index": {"Finding": [0]}}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        state = ResearchTools(replace(settings, max_search_queries=1), EvidenceStore(tmp_path), client=client)
        assert "S1" in state.search_papers("gait")
        assert "budget exceeded" in state.search_papers("other")
        assert (tmp_path / "raw_results.jsonl").exists()
        assert len(state.store.sources) == 1


def test_capability_and_out_of_scope_routes_no_model(settings):
    workflow = ResearchWorkflow(settings)
    assert workflow.classify("What can you do?") == "capabilities"
    assert workflow.classify("Write me a poem") == "out_of_scope"
    result = workflow.run("How can you help me?")
    assert result.route == "capabilities" and not result.sources


def test_llm_classifier_parsing(settings, monkeypatch):
    workflow = ResearchWorkflow(settings)
    monkeypatch.setattr(workflow, "_invoke", lambda **kwargs: '```json\n{"category":"academic"}\n```')
    assert workflow.classify("Find peer-reviewed literature on gait analysis") == "academic"
    assert _extract_json('garbage') == {}


def test_evidence_selection_diversifies_queries(tmp_path):
    store = EvidenceStore(tmp_path)
    items = [
        Source("", "tavily", f"First {i}", f"https://first.example/{i}", "text", [], query="first")
        for i in range(5)
    ] + [
        Source("", "tavily", f"Second {i}", f"https://second.example/{i}", "text", [], query="second")
        for i in range(5)
    ]
    store.add(items)
    assert [source.query for source in store.select(4)] == ["first", "second", "first", "second"]


def test_second_round_resets_per_round_budget(tmp_path, settings):
    def handler(req):
        return httpx.Response(200, json={"results": []})
    cfg = replace(settings, max_search_queries=1, max_research_rounds=2)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        tools = ResearchTools(cfg, EvidenceStore(tmp_path), client=client)
        tools.start_round()
        tools.search_web("first")
        assert "budget exceeded" in tools.search_web("excess")
        tools.start_round()
        assert "error" not in tools.search_web("second")
        assert tools.used_queries == 2
