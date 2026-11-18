"""Regressions for the saved six-citation report and contextual follow-up policy."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from agent import tools
from agent.evidence import parse_citations
from conftest import REPORT, call


# The failing trace retrieved eight records, cited six, then displayed only five
# references after Python clipped S8. Repeated mentions do not consume new slots.
FIVE = ("## Perspective 1\nFirst finding [S1, S3].\n"
        "## Perspective 2\nSecond finding [S5]. Another finding [S6].\n"
        "## Perspective 3\nThird finding [S7]. Repeated support [S1, S3].")
SIX = FIVE + "\n## Overall synthesis\nAdditional evidence [S8]."
START = [call("select_research_route", category="general"), call("read_research_instructions"),
         call("search_web", query="diagnosis trends")]


@pytest.fixture
def eight_sources(monkeypatch):
    monkeypatch.setattr(tools, "tavily_search", lambda *args: [
        {"title": f"Source {i}", "url": f"https://example.org/{i}",
         "content": "The trial included 40 adults with stroke."} for i in range(1, 9)
    ])


def test_eight_retrieved_five_distinct_cited_is_a_valid_five_source_report(make_session, eight_sources):
    workflow, model, _, _, statuses = make_session(START + [FIVE])
    workflow.config = replace(workflow.config, max_sources=5, search_results_per_query=8)
    result = workflow.run("why is there a surge in the ADHD diagnoses")
    assert len(workflow.store.sources) == 8
    assert len(result.sources) == 5 and len(model.calls) == 4
    assert not result.incomplete and statuses[-1] == "Research complete"
    assert not any("source limit" in warning for warning in result.warnings)


def test_actual_six_citations_use_existing_single_correction_not_a_second_stage(make_session, eight_sources):
    workflow, model, _, _, statuses = make_session(START + [SIX, FIVE])
    workflow.config = replace(workflow.config, max_sources=5, search_results_per_query=8)
    result = workflow.run("why is there a surge in the ADHD diagnoses")
    assert len(model.calls) == 5
    assert statuses.count("Correcting answer validation") == 1
    assert not result.incomplete and len(result.sources) == 5
    assert "6" in str(model.calls[-1]["messages"]) and "5" in str(model.calls[-1]["messages"])
    assert len(workflow.store.sources) == 8


def test_failed_single_correction_preserves_genuine_citations_and_reports_exact_violation(make_session, eight_sources):
    workflow, model, _, events, statuses = make_session(START + [SIX, SIX])
    workflow.config = replace(workflow.config, max_sources=5, search_results_per_query=8)
    result = workflow.run("why is there a surge in the ADHD diagnoses")
    assert len(model.calls) == 5 and result.incomplete
    assert statuses.count("Correcting answer validation") == 1
    assert statuses[-1] == "Partial report available"
    assert "[S8]" in result.answer and "[UNVERIFIED]" not in result.answer
    # A rejected draft must remain traceable; it is not a compliant final report.
    assert {s.source_id for s in result.sources} == {"S1", "S3", "S5", "S6", "S7", "S8"}
    assert any("6" in warning and "5" in warning for warning in result.warnings)
    assert "source limit" in result.answer.lower()
    saved = json.loads(Path(result.run_directory, "sources.json").read_text())
    assert len(saved) == 6 and workflow.store.original_result("S8") is not None
    assert all(c.status != "unsupported" for c in result.grounding)


def test_unknown_id_is_still_rejected_without_counting_it_as_retrieved_evidence(make_session, eight_sources):
    bad = FIVE.replace("[S5]", "[S5, S999]")
    workflow, _, _, _, _ = make_session(START + [bad, bad])
    workflow.config = replace(workflow.config, max_sources=5, search_results_per_query=8)
    result = workflow.run("Research diagnosis trends")
    assert result.incomplete and "[S999]" not in result.answer and "[UNVERIFIED]" in result.answer
    assert len(result.sources) == 5
    assert not any("source limit" in warning for warning in result.warnings)
    assert any("S999" in check.citations for check in result.grounding)


def test_follow_up_uses_direct_policy_same_agent_and_can_search_autonomously(make_session):
    workflow, model, providers, events, _ = make_session([
        *START, REPORT,
        call("select_research_route", category="general", follow_up=True),
        call("search_web", query="symptom management"),
        "For your follow-up, start with the approach described in [S1].",
    ])
    first = workflow.run("why is there a surge in the ADHD diagnoses")
    agent = workflow.agent
    second = workflow.run("how can I manage my ADHD symptoms then")
    assert workflow.agent is agent and not second.incomplete
    assert len(providers) == 2
    assert "Response format: direct_answer" in model.calls[-1]["system_prompt"]
    assert "Do not repeat the three-perspective report" in model.calls[-1]["system_prompt"]
    assert "why is there a surge" in str(model.calls[-1]["messages"])
    routes = [e["payload"]["result"] for e in events if e["kind"] == "tool_result"
              and e["payload"]["tool"]["name"] == "select_research_route"]
    assert '"response_format": "direct_answer"' in routes[-1]["content"][0]["text"]
    assert parse_citations(second.answer)[0] == "[S1]"


def test_explicit_full_report_follow_up_keeps_three_perspective_validation(make_session):
    workflow, model, _, _, statuses = make_session([
        *START, REPORT,
        call("select_research_route", category="general", follow_up=True, full_report=True),
        "One brief answer [S1].", REPORT,
    ])
    workflow.run("Research diagnosis trends")
    result = workflow.run("Write a full three-perspective report on those findings")
    assert not result.incomplete
    assert "Response format: research_report" in model.calls[-1]["system_prompt"]
    assert statuses.count("Correcting answer validation") == 1


def test_new_topic_restores_report_policy_and_clears_follow_up_policy(make_session):
    workflow, model, _, _, _ = make_session([
        *START, REPORT,
        call("select_research_route", category="general", follow_up=True), "A direct follow-up [S1].",
        call("select_research_route", category="general"), REPORT,
    ])
    workflow.run("Research diagnosis trends")
    workflow.run("Explain those findings")
    result = workflow.run("New topic: research renewable energy")
    assert not result.incomplete
    assert "Response format: research_report" in model.calls[-1]["system_prompt"]
    assert not workflow._follow_up
