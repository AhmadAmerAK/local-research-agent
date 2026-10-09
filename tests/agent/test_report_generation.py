"""Report-writing regressions: inline evidence, no auxiliary tools, bounded retry."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from agent.config import Settings
from agent.evidence import Source
from agent import orchestrator
from agent.orchestrator import ReportGenerationError, ResearchWorkflow


@pytest.fixture
def settings():
    return Settings("", "", "http://localhost:11434", "qwen3:4b-instruct",
                    2, 3, 1, 3, 8192, True)


def test_report_agent_disables_auxiliary_tools_and_preserves_search_config(monkeypatch, settings):
    harness = pytest.importorskip("strands_harness")
    captured = []
    monkeypatch.setattr(harness, "create_harness", lambda **kwargs: captured.append(kwargs))
    monkeypatch.setattr(ResearchWorkflow, "_model", lambda self, output_tokens: "test-model")
    workflow = ResearchWorkflow(settings)
    workflow._agent(tools=["search_papers"], instructions="search")
    workflow._agent(tools=[], instructions="report", report_only=True)
    search, report = captured
    assert search["tools"] == ["search_papers"]
    assert search["skills"] == [str(orchestrator.AGENT_DIR / "guide" / "skills")]
    assert search["context_manager"] == "auto"
    assert report["tools"] == []
    assert report["skills"] is False
    assert report["context_manager"] == {"stash": {"retrieval_tool": False}}
    assert report["builtin_tools"] == [] and report["builtin_plugins"] == []


def test_real_report_harness_has_no_tools(settings):
    pytest.importorskip("strands_harness")
    workflow = ResearchWorkflow(settings)
    agent = workflow._agent(tools=[], instructions="Write from supplied evidence.", report_only=True)
    try:
        assert agent.tool_names == []
    finally:
        agent.shutdown()


@pytest.fixture
def run_report(monkeypatch, tmp_path, settings):
    guide = orchestrator._read_guide()
    monkeypatch.setattr(orchestrator, "AGENT_DIR", tmp_path)
    monkeypatch.setattr(orchestrator, "_read_guide", lambda: guide)

    def run(answers):
        calls, statuses = [], []
        workflow = ResearchWorkflow(settings, on_status=statuses.append)
        monkeypatch.setattr(workflow, "classify", lambda question: "academic")

        def search(route, question, tools, round_number):
            tools.store.add([
                Source("", "openalex", f"Study {i}", f"https://example.org/{i}",
                       "Retrieved evidence", []) for i in range(4)
            ])

        monkeypatch.setattr(workflow, "_search_round", search)
        remaining = iter(answers)

        def invoke(**kwargs):
            calls.append(kwargs)
            return next(remaining)

        monkeypatch.setattr(workflow, "_invoke", invoke)
        return workflow, calls, statuses, tmp_path

    return run


def test_successful_report_uses_guide_and_does_not_retry(run_report):
    workflow, calls, statuses, _ = run_report([
        "## Perspective 1\nEvidence [S1].\n## Perspective 2\nEvidence [S2]."
        "\n## Perspective 3\nEvidence [S3]."
    ])
    result = workflow.run("Why are diagnoses increasing?")
    assert len(calls) == 1
    assert "three distinct" in calls[0]["instructions"].lower()
    assert "COMPLETE report" in calls[0]["prompt"]
    assert "Perspective 1, Perspective 2, Perspective 3" in calls[0]["prompt"]
    assert calls[0]["report_only"] is True
    assert calls[0]["output_tokens"] == 2048
    assert "## References" in result.answer and "https://example.org/0" in result.answer
    assert statuses[-1] == "Research complete"


@pytest.mark.parametrize("first_answer", [
    "Let’s begin with the synthesis.",
    "An invented reference [S999].",
    "A source excluded from supplied evidence [S4].",
    "Let’s begin.\n## References\n[S1] Study 0",
])
def test_citation_free_or_invalid_answer_retries_once(run_report, first_answer):
    workflow, calls, statuses, _ = run_report([first_answer, "A completed report [S1] [S2] [S3]."])
    result = workflow.run("Why are diagnoses increasing?")
    assert len(calls) == 2
    assert "CORRECTION" in calls[1]["prompt"]
    assert all(call["report_only"] for call in calls)
    assert any("Retrying" in status for status in statuses)
    assert "A completed report" in result.answer
    assert statuses[-1] == "Research complete"


def test_two_uncited_answers_fail_and_preserve_evidence(run_report):
    workflow, calls, statuses, directory = run_report(["I will start.", "Let’s begin."])
    with pytest.raises(ReportGenerationError, match="Report generation failed"):
        workflow.run("Why are diagnoses increasing?")
    assert len(calls) == 2
    assert statuses[-1] == "Report generation failed"
    assert "Research complete" not in statuses
    run_dir = next((directory / "memory" / "runs").iterdir())
    assert (run_dir / "sources.json").is_file()
    assert not (run_dir / "report.md").exists()
    assert "Report generation failed" in json.loads((run_dir / "warnings.json").read_text())[0]


@pytest.mark.parametrize("continuation_error", [None, "limit", "connection"])
def test_output_limit_continues_same_agent_once_and_retains_text(monkeypatch, settings, continuation_error):
    exceptions = pytest.importorskip("strands.types.exceptions")
    limit = exceptions.MaxTokensReachedException
    streamed, statuses, created = [], [], []
    workflow = ResearchWorkflow(settings, on_text=streamed.append, on_status=statuses.append)

    class FakeAgent:
        def __init__(self):
            self.callback_handler = lambda **event: workflow.on_text(event["data"])
            self.messages = []
            self.prompts = []
            self.closed = False

        def __call__(self, prompt):
            self.prompts.append(prompt)
            chunk = "Evidence [S1].\n" if len(self.prompts) == 1 else "Further evidence [S2]."
            self.callback_handler(data=chunk)
            self.messages.append({"role": "assistant", "content": [{"text": chunk}]})
            if len(self.prompts) == 1 or continuation_error == "limit":
                raise limit("Maximum token limit")
            if continuation_error == "connection":
                raise ConnectionError("Connection interrupted")
            return chunk

        def shutdown(self):
            self.closed = True

    agent = FakeAgent()

    def create(**kwargs):
        created.append(kwargs)
        return agent

    monkeypatch.setattr(workflow, "_agent", create)
    output = workflow._invoke(prompt="Supplied evidence", report_only=True, stream_text=True)
    assert len(created) == 1 and len(agent.prompts) == 2
    assert "exactly where" in agent.prompts[1]
    assert output == "Evidence [S1].\nFurther evidence [S2]."
    assert "".join(streamed) == output
    assert agent.closed
    assert statuses == ["Continuing report generation"]
    assert bool(workflow._report_warning) is (continuation_error is not None)


def test_output_limit_in_search_is_not_automatically_continued(monkeypatch, settings):
    exceptions = pytest.importorskip("strands.types.exceptions")
    class FakeAgent:
        closed = False
        def __call__(self, prompt):
            raise exceptions.MaxTokensReachedException("Maximum token limit")
        def shutdown(self):
            self.closed = True
    agent = FakeAgent()
    workflow = ResearchWorkflow(settings)
    monkeypatch.setattr(workflow, "_agent", lambda **kwargs: agent)
    with pytest.raises(exceptions.MaxTokensReachedException):
        workflow._invoke(prompt="Search")
    assert agent.closed
