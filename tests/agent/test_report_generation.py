"""Final answer validation and continuation through the native research loop."""
import json
from pathlib import Path

import pytest

from conftest import REPORT, call
from agent import tools
from agent.orchestrator import CORE_INSTRUCTIONS, ReportGenerationError


START = [call("select_research_route", category="academic"), call("read_research_instructions"),
         call("search_papers", query="clinical evidence")]


def test_successful_report_retains_verified_references(make_session):
    workflow, model, _, _, statuses = make_session(START + [REPORT])
    result = workflow.run("Research clinical evidence")
    assert not result.incomplete and len(result.sources) == 3
    assert "## References" in result.answer
    assert statuses[-1] == "Research complete"
    assert len(model.calls) == 4
    assert workflow.agent.messages[-1]["content"][0]["text"] == result.answer


@pytest.mark.parametrize("invalid", [
    "Let’s begin with the synthesis.",
    "An invented reference [S999].",
    "A report with one source [S1].",
    "All sources but no perspectives [S1, S2, S3].",
    REPORT.replace("Practical evidence [S2].", "Practical evidence."),
    REPORT.replace("[S1]", "[S1, S999]"),
    "Let’s begin.\n**References**\n[S1] Fabricated bibliography.",
])
def test_native_validation_hook_requests_one_correction(make_session, invalid):
    workflow, model, _, events, statuses = make_session(START + [invalid, REPORT])
    result = workflow.run("Research clinical methods")
    assert not result.incomplete
    assert len(model.calls) == 5
    assert "Correct these answer issues" in str(model.calls[-1]["messages"])
    assert statuses.count("Correcting answer validation") == 1
    assert "[S999]" not in result.answer


def test_report_with_invalid_group_after_retry_is_preserved_as_partial(make_session):
    malformed = REPORT.replace("[S1]", "[S1, S999]")
    workflow, model, _, _, statuses = make_session(START + [malformed, malformed])
    result = workflow.run("Research clinical methods")
    assert result.incomplete and "[UNVERIFIED]" in result.answer
    assert "[S999]" not in result.answer
    assert any("[S999]" in warning for warning in result.warnings)
    assert statuses[-1] == "Partial report available"
    assert Path(result.run_directory, "report.md").read_text().strip() == result.answer
    assert json.loads(Path(result.run_directory, "warnings.json").read_text()) == result.warnings


def test_two_uncited_answers_fail_without_new_agent_or_research_loop(make_session):
    workflow, model, providers, _, statuses = make_session(START + ["I will start.", "Let’s begin."])
    with pytest.raises(ReportGenerationError, match="no valid evidence citations"):
        workflow.run("Research clinical methods")
    assert len(model.calls) == 5 and len(providers) == 1
    assert (workflow.store.directory / "sources.json").is_file()
    assert "Research complete" not in statuses


def test_grouped_citations_have_one_reference_per_source(make_session):
    workflow, _, _, _, _ = make_session(START + [REPORT.replace("[S1]", "[S1, S2]")])
    result = workflow.run("Research clinical methods")
    assert "[S1] [S2]" in result.answer and not result.incomplete
    for source in result.sources:
        assert result.answer.count(f"- [{source.source_id}]") == 1


def test_insufficient_evidence_does_not_trigger_pointless_regeneration(make_session, monkeypatch):
    monkeypatch.setattr(tools, "openalex_search", lambda *args: [{
        "id": "https://openalex.org/W1", "title": "One study", "abstract_inverted_index": {"Finding": [0]},
    }])
    workflow, model, _, _, statuses = make_session(START + ["Available finding [S1]. Evidence is limited."])
    result = workflow.run("Research clinical methods")
    assert result.incomplete and len(model.calls) == 4
    assert any("insufficient evidence" in warning for warning in result.warnings)
    assert statuses[-1] == "Partial report available"


@pytest.mark.parametrize("ending", ["complete", "limit", "connection"])
def test_continuation_uses_same_agent_and_preserves_partial_text(make_session, ending):
    partial = "## Perspective 1\nFirst findings [S1].\n"
    rest = "## Perspective 2\nSecond findings [S2].\n## Perspective 3\nThird findings [S3]."
    next_step = rest if ending == "complete" else {"text": rest, "limit": True} if ending == "limit" else ConnectionError("Interrupted")
    workflow, model, providers, events, statuses = make_session(START + [{"text": partial, "limit": True}, next_step])
    result = workflow.run("Research clinical methods")
    assert "First findings" in result.answer
    assert len(model.calls) == 5 and len(providers) == 1
    assert statuses.count("Continuing answer generation") == 1
    assert result.incomplete is (ending != "complete")
    if ending != "connection":
        assert "Third findings" in result.answer
    assert workflow.agent is not None  # It lives until session close, not end of turn.


def test_persistent_agent_keeps_skills_lazy_and_only_current_provider_exposed(make_session, monkeypatch):
    workflow, model, _, _, _ = make_session([])
    workflow._ensure_session()
    assert set(workflow.agent.tool_names) == {
        "select_research_route", "read_research_instructions", "read_evidence",
    }
    assert "# Research Agent" not in workflow.agent.system_prompt
    assert "CORE_INSTRUCTIONS" not in workflow.agent.system_prompt
    assert workflow.agent._context_manager_instance is not None
