"""Operational settings must reach the model, validation and native tool budgets."""
from dataclasses import replace
import json

import httpx
import pytest

from agent.config import Settings
from agent.evidence import EvidenceStore
from agent.orchestrator import ResearchWorkflow
from agent.report_validation import validate_report
from agent.tools import ResearchTools, openalex_search, tavily_search
from conftest import REPORT, call


NUMERIC_SETTINGS = (
    "MAX_SEARCH_QUERIES", "MAX_RESEARCH_ROUNDS", "MAX_SOURCES", "MIN_REQUIRED_SOURCES",
    "CONTEXT_WINDOW", "OUTPUT_TOKENS", "AGENT_MAX_TURNS", "MAX_QUERY_CHARS",
    "SEARCH_RESULTS_PER_QUERY", "SEARCH_SNIPPET_CHARS", "EVIDENCE_PAGE_CHARS",
    "EVIDENCE_CATALOGUE_PAGE_SIZE",
)


@pytest.fixture
def configured(monkeypatch):
    for name in NUMERIC_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    return Settings.from_env


def test_existing_defaults_and_derived_budgets(configured):
    settings = configured()
    assert settings.context_window == 8192
    assert settings.output_tokens == 3072
    assert settings.search_budget == 2
    assert settings.agent_turn_budget == 14
    assert settings.results_per_query == settings.max_sources == 5


def test_previous_ceilings_are_not_hardware_constraints(configured, monkeypatch):
    values = {"MAX_SEARCH_QUERIES": "8", "MAX_RESEARCH_ROUNDS": "5", "MAX_SOURCES": "50",
              "MIN_REQUIRED_SOURCES": "4", "CONTEXT_WINDOW": "65536", "OUTPUT_TOKENS": "4096"}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    settings = configured()
    assert settings.search_budget == 40 and settings.agent_turn_budget == 52
    assert settings.max_sources == 50 and settings.min_required_sources == 4
    assert settings.context_window == 65536 and settings.output_tokens == 4096


@pytest.mark.parametrize("name", NUMERIC_SETTINGS)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "1.5", "invalid"])
def test_budgets_require_positive_finite_integers(configured, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        configured()


def test_blank_settings_use_documented_defaults(configured, monkeypatch):
    for name in NUMERIC_SETTINGS:
        monkeypatch.setenv(name, "  ")
    settings = configured()
    assert settings.output_tokens == 3072 and settings.agent_turn_budget == 14


def test_small_positive_budgets_are_allowed_when_consistent(configured, monkeypatch):
    for name in NUMERIC_SETTINGS:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CONTEXT_WINDOW", "2")
    settings = configured()
    assert settings.search_budget == settings.agent_turn_budget == settings.results_per_query == 1


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_direct_settings_construction_cannot_bypass_budget_validation(configured, value):
    with pytest.raises(ValueError, match="MAX_SEARCH_QUERIES"):
        replace(configured(), max_search_queries=value)


def test_source_minimum_cannot_exceed_report_maximum(configured, monkeypatch):
    monkeypatch.setenv("MIN_REQUIRED_SOURCES", "6")
    with pytest.raises(ValueError, match="MIN_REQUIRED_SOURCES cannot exceed MAX_SOURCES"):
        configured()


@pytest.mark.parametrize("output", ["8192", "8193"])
def test_output_budget_must_leave_room_for_input(configured, monkeypatch, output):
    monkeypatch.setenv("OUTPUT_TOKENS", output)
    with pytest.raises(ValueError, match="OUTPUT_TOKENS must be less than CONTEXT_WINDOW"):
        configured()


def test_configured_output_and_context_reach_ollama_request(configured, monkeypatch):
    monkeypatch.setenv("CONTEXT_WINDOW", "16384")
    monkeypatch.setenv("OUTPUT_TOKENS", "4096")
    workflow = ResearchWorkflow(configured())
    request = workflow._model().format_request([{"role": "user", "content": [{"text": "Research question"}]}])
    assert request["options"]["num_ctx"] == 16384
    assert request["options"]["num_predict"] == 4096
    assert request["options"]["temperature"] == .15
    assert request["keep_alive"] == "5m"


@pytest.mark.parametrize("minimum", [1, 2, 4])
def test_cited_source_count_uses_setting_independently_of_three_perspectives(minimum):
    report = REPORT.replace("S2", "S1").replace("S3", "S1")
    if minimum > 1:
        report += "\nAdditional evidence " + " ".join(f"[S{i}]" for i in range(2, minimum + 1)) + "."
    assert validate_report(report, {f"S{i}" for i in range(1, minimum + 1)}, minimum) == []
    assert "Missing Perspective 2 section." in validate_report("One finding [S1].", {"S1"}, 1)


def test_minimum_applies_to_cited_sources_not_just_retrieved_sources():
    issues = validate_report(REPORT, {"S1", "S2", "S3", "S4"}, 4)
    assert any("4 distinct valid source citations; found 3" in issue for issue in issues)


def test_one_source_can_support_three_perspectives_without_retry(make_session):
    report = REPORT.replace("S2", "S1").replace("S3", "S1")
    workflow, model, _, _, _ = make_session([
        call("select_research_route", category="academic"), call("read_research_instructions"),
        call("search_papers", query="clinical methods"), report,
    ])
    workflow.config = replace(workflow.config, min_required_sources=1)
    result = workflow.run("Research clinical methods")
    assert not result.incomplete and len(result.sources) == 1 and len(model.calls) == 4


def test_explicit_native_loop_budget_stops_without_invalid_history(make_session):
    workflow, model, _, _, _ = make_session([
        call("select_research_route", category="academic"), call("read_research_instructions"),
        *[call("read_evidence") for _ in range(20)],
    ])
    workflow.config = replace(workflow.config, agent_max_turns=3)
    result = workflow.run("Research clinical methods")
    assert len(model.calls) == 3 and not result.sources
    assert workflow._partial_warning == "Agent stopped: limit_turns."


def test_correction_retry_cannot_bypass_model_call_budget(make_session):
    from agent.orchestrator import ReportGenerationError
    workflow, model, _, _, _ = make_session([
        call("select_research_route", category="academic"), call("read_research_instructions"),
        call("search_papers", query="clinical methods"), "An uncited answer.",
        call("read_evidence"), REPORT,
    ])
    workflow.config = replace(workflow.config, agent_max_turns=5)
    with pytest.raises(ReportGenerationError, match="no valid evidence citations"):
        workflow.run("Research clinical methods")
    assert len(model.calls) == 5 and workflow._model_calls == 5
    assert model.steps == [REPORT]


def test_token_continuation_cannot_bypass_total_turn_budget(make_session):
    workflow, model, _, _, _ = make_session([
        call("select_research_route", category="academic"), call("read_research_instructions"),
        call("search_papers", query="clinical methods"), {"text": REPORT, "limit": True},
    ])
    workflow.config = replace(workflow.config, agent_max_turns=4)
    result = workflow.run("Research clinical methods")
    assert len(model.calls) == 4 and result.incomplete
    assert any("turn budget" in warning for warning in result.warnings)


def test_long_query_is_not_silently_truncated_and_rejection_spends_no_budget(configured, tmp_path):
    settings = replace(configured(), max_query_chars=400)
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"results": []})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        research = ResearchTools(settings, EvidenceStore(tmp_path), client=client)
        assert "error" not in json.loads(research.search_papers("x" * 400))
        assert requests[0].url.params["search"] == "x" * 400
        error = json.loads(research.search_papers("x" * 401))["error"]
        assert "400 characters" in error and "shorten" in error.lower()
        assert len(requests) == 1 and research.used_queries == 1


def test_native_agent_can_refine_a_rejected_query_without_spending_search_budget(make_session):
    workflow, _, providers, events, _ = make_session([
        call("select_research_route", category="academic"), call("read_research_instructions"),
        call("search_papers", query="x" * 401), call("search_papers", query="x" * 400), REPORT,
    ], searches=1, rounds=1)
    workflow.config = replace(workflow.config, max_query_chars=400)
    result = workflow.run("Research clinical methods")
    assert not result.incomplete and providers == [("openalex", "x" * 400)]
    assert workflow.search.used_queries == 1
    rejected = [e["payload"]["result"] for e in events if e["kind"] == "tool_result"
                and e["payload"]["tool"]["name"] == "search_papers"][0]
    assert rejected["status"] == "error" and "shorten" in str(rejected)


def test_search_budget_is_finite_after_removing_old_ceiling(configured, tmp_path):
    settings = replace(configured(), max_search_queries=6, max_research_rounds=5)
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"results": []}))) as client:
        research = ResearchTools(settings, EvidenceStore(tmp_path), client=client)
        research.enforce_route = True
        research.active_route = "academic"
        for index in range(30):
            assert "error" not in json.loads(research.search_papers(f"query {index}"))
        assert "budget exceeded" in research.search_papers("one too many").lower()
        assert research.used_queries == 30


@pytest.mark.parametrize(("provider", "cap"), [("openalex", 100), ("tavily", 20)])
def test_only_documented_provider_page_caps_remain(provider, cap):
    requests = []
    with httpx.Client(transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200, json={"results": []}))) as client:
        search = openalex_search if provider == "openalex" else tavily_search
        search("query", "fake-key", count=500, client=client)
    if provider == "openalex":
        assert int(requests[0].url.params["per_page"]) == cap
    else:
        assert json.loads(requests[0].content)["max_results"] == cap


def test_results_per_query_is_independent_of_report_source_cap(configured, monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_RESULTS_PER_QUERY", "30")
    monkeypatch.setenv("MAX_SOURCES", "4")
    settings = configured()
    def respond(request):
        assert request.url.params["per_page"] == "30"
        return httpx.Response(200, json={"results": []})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        research = ResearchTools(settings, EvidenceStore(tmp_path), client=client)
        payload = json.loads(research.search_papers("query"))
    assert settings.max_sources == 4
    assert payload["requested_results"] == payload["result_limit"] == 30


def test_configured_preview_and_pages_retain_access_to_all_original_evidence(make_session, monkeypatch):
    from agent import tools
    original = [{"id": f"https://openalex.org/W{i}", "title": f"Study {i}",
                 "abstract_inverted_index": {"Evidence": list(range(1000))}}
                for i in range(1, 4)]
    monkeypatch.setattr(tools, "openalex_search", lambda *args: original)
    workflow, _, _, events, _ = make_session([
        call("select_research_route", category="academic"), call("read_research_instructions"),
        call("search_papers", query="clinical methods"), REPORT,
        call("select_research_route", category="academic", follow_up=True),
        call("read_evidence"), call("read_evidence", offset=2),
        call("read_evidence", source_id="S1"), call("read_evidence", source_id="S1", offset=37),
        "Prior evidence [S1].",
    ])
    workflow.config = replace(workflow.config, search_snippet_chars=2500, evidence_page_chars=37,
                              evidence_catalogue_page_size=2)
    workflow.run("Research clinical methods")
    workflow.run("Revisit the original evidence")
    results = [(event["payload"]["tool"]["name"], json.loads(event["payload"]["result"]["content"][0]["text"]))
               for event in events if event["kind"] == "tool_result"
               and event["payload"]["tool"]["name"] in {"search_papers", "read_evidence"}]
    search = results[0][1]
    assert search["snippet_char_limit"] == 2500
    assert all(len(source["snippet"]) == 2500 for source in search["new_sources"])
    catalogue1, catalogue2, page1, page2 = [result for _, result in results[1:]]
    assert [s["id"] for s in catalogue1["sources"]] == ["S1", "S2"]
    assert catalogue1["next_offset"] == 2
    assert [s["id"] for s in catalogue2["sources"]] == ["S3"] and catalogue2["next_offset"] is None
    assert page1["next_offset"] == 37 and page2["next_offset"] == 74
    raw_json = json.dumps(original[0], ensure_ascii=False)
    assert page1["original_result"] + page2["original_result"] == raw_json[:74]
    assert workflow.store.original_result("S1") == original[0]


def test_evidence_limits_never_truncate_citation_identifiers(tmp_path):
    from agent.evidence import normalise_tavily
    prefix = "https://example.org/" + "a" * 700
    raw = [{"title": "Study", "url": prefix + ending, "content": "Evidence " * 1000}
           for ending in ("one", "two")]
    store = EvidenceStore(tmp_path)
    store.save_raw("tavily", "query", raw)
    store.add([normalise_tavily(item, snippet_chars=37) for item in raw])
    assert len(store.sources) == 2
    assert store.sources[0].url == raw[0]["url"]
    assert len(store.sources[0].snippet) == 37
    assert store.original_result("S2") == raw[1]
