"""Session behaviour is exercised through the installed native Strands loop."""
import json
from dataclasses import replace

import pytest

from conftest import REPORT, call
from agent import tools
from agent.orchestrator import ReportGenerationError


START = [call("select_research_route", category="academic"), call("read_research_instructions")]


def test_native_loop_researches_refines_and_answers_with_one_agent(make_session):
    workflow, model, providers, events, statuses = make_session(START + [
        call("search_papers", query="initial evidence"),
        call("search_papers", query="contradictory findings and gaps"), REPORT,
    ])
    result = workflow.run("Research clinical methods")
    assert providers == [("openalex", "initial evidence"), ("openalex", "contradictory findings and gaps")]
    assert result.incomplete is False and len(result.sources) == 3
    assert "search_papers" not in model.calls[0]["tools"]
    assert "search_papers" in model.calls[1]["tools"]
    assert all("search_web" not in request["tools"] for request in model.calls)
    assert len([event for event in events if event["kind"] == "tool_call"]) == 4
    assert statuses[-1] == "Research complete"


def test_follow_up_reuses_same_agent_history_evidence_and_ids(make_session):
    workflow, model, providers, events, statuses = make_session(START + [
        call("search_papers", query="clinical methods"), REPORT,
        call("select_research_route", category="academic", follow_up=True),
        call("read_evidence", source_id="S1"), "The first approach is suitable based on [S1].",
    ])
    first = workflow.run("Research clinical methods")
    agent = workflow.agent
    second = workflow.run("Which of those is suitable for rehabilitation?")
    assert workflow.agent is agent
    assert len(providers) == 1
    assert second.sources[0].source_id == first.sources[0].source_id == "S1"
    assert not second.incomplete
    assert "search_papers" not in model.calls[4]["tools"]  # Re-authorize each user turn.
    history = str(model.calls[-1]["messages"])
    assert "Research clinical methods" in history and "Which of those" in history
    assert "Original" in history
    assert len(json.loads((workflow.store.directory / "conversation.json").read_text())) == 2


def test_new_sessions_have_separate_agents_context_and_evidence(make_session):
    first, _, _, _, _ = make_session(START + [call("search_papers", query="clinical"), REPORT])
    second, model, _, _, _ = make_session([
        call("select_research_route", category="general"), call("read_research_instructions"),
        call("search_web", query="energy policy"), REPORT,
    ])
    first.run("Private clinical question")
    second.run("Research energy policy")
    assert first.agent is not second.agent
    assert first.store.directory != second.store.directory
    assert "Private clinical question" not in str(model.calls)
    assert all(source.provider == "tavily" for source in second.store.sources)
    assert second.store.sources[0].source_id == "S1"


def test_topic_change_removes_old_provider_and_blocks_forged_calls(make_session):
    workflow, model, providers, events, _ = make_session(START + [
        call("search_papers", query="clinical"), REPORT,
        call("search_papers", query="stale permissions"),
        call("select_research_route", category="general"),
        call("search_papers", query="wrong provider"),
        call("search_web", query="energy policy"), REPORT.replace("S1", "S4").replace("S2", "S5").replace("S3", "S6"),
    ])
    workflow.run("Research clinical methods")
    result = workflow.run("Now research energy policy")
    assert providers == [("openalex", "clinical"), ("tavily", "energy policy")]
    assert result.route == "general" and all(source.provider == "tavily" for source in result.sources)
    assert all("search_papers" not in request["tools"] for request in model.calls[4:])
    assert any(event["kind"] == "tool_result" and event["payload"]["result"]["status"] == "error" for event in events)


def test_budget_enforced_and_renewed_per_question_without_new_ids(make_session):
    workflow, model, providers, _, _ = make_session(START + [
        call("search_papers", query="first"), call("search_papers", query="over budget"), REPORT,
        call("select_research_route", category="academic", follow_up=True),
        call("search_papers", query="second turn"), "Follow-up evidence [S1].",
    ], searches=1, rounds=1)
    workflow.run("Research methods")
    workflow.run("Investigate those further")
    assert providers == [("openalex", "first"), ("openalex", "second turn")]
    assert len(workflow.store.sources) == 3
    assert workflow.search.used_queries == 1


def test_route_cannot_be_switched_mid_turn(make_session):
    workflow, _, providers, events, _ = make_session(START + [
        call("select_research_route", category="general"), call("search_web", query="bypass"),
        call("search_papers", query="permitted"), REPORT,
    ])
    workflow.run("Research clinical methods")
    assert providers == [("openalex", "permitted")]
    assert workflow.route == "academic"


def test_evidence_survives_context_reduction_and_original_record_can_be_recovered(make_session):
    workflow, model, providers, events, _ = make_session(START + [
        call("search_papers", query="clinical"), REPORT,
        call("select_research_route", category="academic", follow_up=True),
        call("read_evidence"), call("read_evidence", source_id="S1"), "Recovered findings [S1].",
    ])
    workflow.run("Research clinical methods")
    # Simulate history having been reduced; original evidence is independent of messages.
    workflow.agent.messages[:] = [{"role": "user", "content": [{"text": "Summary: clinical methods; source S1."}]}]
    result = workflow.run("Check the original abstract of S1")
    assert not result.incomplete and len(providers) == 1
    recovered = [e["payload"]["result"] for e in events if e["kind"] == "tool_result"
                 and e["payload"]["tool"]["name"] == "read_evidence"][-1]
    assert "abstract_inverted_index" in str(recovered)
    assert workflow.agent._context_manager_instance is not None


def test_progressive_guide_loading_is_enforced_before_http(make_session):
    workflow, model, providers, events, _ = make_session([
        call("select_research_route", category="academic"),
        call("search_papers", query="premature"), call("read_research_instructions"),
        call("search_papers", query="authorised"), REPORT,
    ])
    workflow.run("Research methods")
    assert providers == [("openalex", "authorised")]
    assert "Never fabricate citations." not in model.calls[0]["system_prompt"]
    assert "Never fabricate citations." in str(model.calls[-1]["messages"])


def test_native_skills_disclose_descriptions_before_requested_content(make_session):
    workflow, model, _, events, _ = make_session(START + [
        call("search_papers", query="clinical"),
        call("skills", skill_name="report-writing"), REPORT,
        call("select_research_route", category="academic", follow_up=True),
        "Follow-up finding [S1].",
    ], skills=True)
    result = workflow.run("Research clinical methods")
    workflow.run("Explain the first finding further")
    assert not result.incomplete
    first_prompt = model.calls[0]["system_prompt"]
    assert "report-writing" in first_prompt
    assert "[Concise synthesis]" not in first_prompt
    assert "[Concise synthesis]" not in str(model.calls[3]["messages"])
    assert "[Concise synthesis]" in str(model.calls[-1]["messages"])
    assert all(request["system_prompt"].count("<available_skills>") == 1 for request in model.calls)
    assert [e["payload"]["name"] for e in events if e["kind"] == "tool_call"].count("skills") == 1


def test_original_evidence_is_readable_in_pages_after_native_context_reduction(make_session, monkeypatch):
    from agent import orchestrator
    long_guide = "Research instructions.\n" + "Full policy detail. " * 2000
    monkeypatch.setattr(orchestrator, "_read_guide", lambda: long_guide)
    original = [{"id": f"https://openalex.org/W{i}", "title": f"Study {i}",
                 "abstract_inverted_index": {"Evidence": [0]},
                 "original_detail": "Detailed evidence. " * 1500} for i in range(1, 4)]
    monkeypatch.setattr(tools, "openalex_search", lambda *args: original)
    workflow, model, _, events, _ = make_session(START + [
        call("search_papers", query="clinical"), REPORT,
        call("select_research_route", category="academic", follow_up=True),
        call("read_evidence", source_id="S1"),
        call("read_evidence", source_id="S1", offset=2000), "Original findings [S1].",
    ])
    workflow.run("Research clinical methods")
    # The real native context manager offloads the oversized guide tool result.
    results = [block["toolResult"] for message in model.calls[2]["messages"]
               for block in message["content"] if "toolResult" in block]
    assert len(str(results[-1])) < len(long_guide)
    result = workflow.run("Check S1 against its original evidence")
    assert not result.incomplete
    reads = [e["payload"]["result"] for e in events if e["kind"] == "tool_result"
             and e["payload"]["tool"]["name"] == "read_evidence"]
    recovered = [json.loads(item["content"][0]["text"]) for item in reads]
    assert recovered[0]["next_offset"] == 2000
    assert recovered[1]["next_offset"] == 4000
    assert recovered[0]["original_result"] + recovered[1]["original_result"] == json.dumps(original[0], ensure_ascii=False)[:4000]


def test_outage_remains_a_single_meaningful_result(make_session, monkeypatch):
    def fail(*args, **kwargs):
        raise tools.ResearchAPIError("Research service unreachable or timed out")
    monkeypatch.setattr(tools, "tavily_search", fail)
    workflow, _, _, _, _ = make_session([
        call("select_research_route", category="general"), call("read_research_instructions"),
        call("search_web", query="energy"), {"text": "The service is unavailable.", "limit": True},
        {"text": "Please retry.", "limit": True},
    ])
    result = workflow.run("Research energy")
    assert not result.sources and "couldn't retrieve sources" in result.answer
    assert "has been kept" not in result.answer


def test_closed_session_cannot_be_reused(make_session):
    workflow, _, _, _, _ = make_session([])
    workflow.run("What can you do?")
    workflow.close()
    workflow.close()
    with pytest.raises(RuntimeError, match="closed"):
        workflow.run("Research methods")


def test_native_loop_limit_stops_repeated_denied_searches(make_session):
    workflow, model, providers, _, _ = make_session(START + [
        call("search_papers", query=f"attempt {i}") for i in range(30)
    ], searches=1, rounds=1)
    with pytest.raises(ReportGenerationError, match="no valid evidence citations"):
        workflow.run("Research methods")
    assert len(model.calls) == 13  # One search plus native loop allowance for session tools.
    assert len(providers) == 1
    assert workflow._partial_warning == "Agent stopped: limit_turns."
    assert len(workflow.store.sources) == 3
    assert len(model.steps) > 0


def test_same_batch_route_and_provider_attempts_cannot_bypass_gate(make_session):
    workflow, _, providers, events, _ = make_session([
        [call("select_research_route", category="academic"), call("search_web", query="bypass")],
        call("read_research_instructions"), call("search_papers", query="clinical"), REPORT,
    ])
    result = workflow.run("Research clinical methods")
    assert not result.incomplete
    assert providers == [("openalex", "clinical")]


def test_failed_instruction_loading_does_not_authorize_research(make_session, monkeypatch):
    from agent import orchestrator
    def unavailable():
        raise OSError("Instructions unavailable")
    monkeypatch.setattr(orchestrator, "_read_guide", unavailable)
    workflow, _, providers, _, _ = make_session(START + [
        call("search_papers", query="premature"),
        "Unable to load instructions.", "Unable to load instructions.",
    ])
    result = workflow.run("Research clinical methods")
    assert not workflow._guide_loaded
    assert providers == [] and not result.sources


def test_configured_minimum_evidence_is_preserved(make_session):
    workflow, _, _, _, _ = make_session(START + [
        call("search_papers", query="clinical"), REPORT, REPORT,
    ])
    workflow.config = replace(workflow.config, min_required_sources=4)
    result = workflow.run("Research clinical methods")
    assert result.incomplete
    assert any("configured minimum is 4" in warning for warning in result.warnings)


def test_loop_limit_preserves_tool_call_pairs_for_next_question(make_session):
    workflow, _, providers, _, _ = make_session(START + [call("read_evidence") for _ in range(30)],
                                               searches=1, rounds=1)
    result = workflow.run("Research clinical methods")
    assert not providers and not result.sources
    calls = {block["toolUse"]["toolUseId"] for message in workflow.agent.messages
             for block in message["content"] if "toolUse" in block}
    results = {block["toolResult"]["toolUseId"] for message in workflow.agent.messages
               for block in message["content"] if "toolResult" in block}
    assert calls == results
    assert workflow.agent.messages[-1]["content"] == [{"text": result.answer}]
