"""Offline synthetic KB contract tests; never contact AWS or Ollama."""
from dataclasses import replace
from pathlib import Path
import json
from unittest.mock import Mock

import botocore.session
from botocore.validate import validate_parameters
import pytest

from agent.config import Settings
from agent.rag import CITATION_FAILURE, INSUFFICIENT, LocalAnswerWorkflow, _estimated_tokens
from conftest import ScriptedModel


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "synthetic-openalex")
    monkeypatch.setenv("TAVILY_API_KEY", "synthetic-tavily")
    monkeypatch.delenv("BEDROCK_NUMBER_OF_RESULTS", raising=False)
    monkeypatch.setenv("AWS_REGION", "eu-west-2")
    monkeypatch.setenv("BEDROCK_KNOWLEDGE_BASE_ID", "FAKEKB1234")
    return Settings.from_env()


def hit(text="Fictional procedure: the quality lead classifies deviations."):
    return {"content": {"type": "TEXT", "text": text}, "score": 0.9,
            "location": {"type": "S3", "s3Location": {"uri": "s3://fictional/procedure.txt"}},
            "metadata": {"title": "Fictional procedure", "_chunk_id": "chunk-1", "private": "excluded"}}


def workflow(config, tmp_path, hits, answer="The quality lead classifies deviations [KB1].", **kwargs):
    client = Mock()
    client.retrieve.return_value = {"retrievalResults": hits}
    model = ScriptedModel([answer])
    return LocalAnswerWorkflow(config, client=client, model=model, output_dir=tmp_path, **kwargs), client, model


@pytest.mark.parametrize("reranking", ["NONE", "MANAGED"])
def test_managed_request_and_grounded_generation(config, tmp_path, reranking):
    flow, client, model = workflow(config, tmp_path, [hit()], reranking=reranking)
    result = flow.run("Who classifies deviations?")
    request = client.retrieve.call_args.kwargs
    assert request == {"knowledgeBaseId": "FAKEKB1234", "retrievalQuery": {"text": "Who classifies deviations?"},
                       "retrievalConfiguration": {"managedSearchConfiguration": {
                           "numberOfResults": 5, "rerankingModelType": reranking}}}
    service = botocore.session.get_session().get_service_model("bedrock-agent-runtime")
    validate_parameters(request, service.operation_model("Retrieve").input_shape)
    assert model.calls[0]["tools"] == []
    assert "untrusted" in model.calls[0]["system_prompt"]
    assert "quality lead" in json.dumps(model.calls[0]["messages"])
    assert "[KB1]" in result.answer and "s3://fictional/procedure.txt" in result.answer
    record = json.loads((tmp_path / result.run_directory / "evidence.json").read_text())
    assert record["chunks"][0]["chunk_id"] == "chunk-1"
    assert "private" not in json.dumps(record)
    assert "content" not in record["chunks"][0]
    flow.close()
    client.close.assert_called_once()


@pytest.mark.parametrize("reranking", ["NONE", "MANAGED"])
def test_configured_limit_requests_and_uses_more_than_five_hits(config, tmp_path, monkeypatch, reranking):
    monkeypatch.setenv("BEDROCK_NUMBER_OF_RESULTS", "8")
    flow, client, model = workflow(replace(config, context_window=32768), tmp_path,
                                  [hit("Fictional evidence.")] * 9,
                                  answer="Fictional evidence [KB8].", reranking=reranking)
    result = flow.run("Question")
    assert client.retrieve.call_args.kwargs["retrievalConfiguration"]["managedSearchConfiguration"] == {
        "numberOfResults": 8, "rerankingModelType": reranking}
    assert len(result.sources) == 8
    assert "[KB8]" in result.answer
    context = json.loads(model.calls[0]["messages"][0]["content"][0]["text"])["untrusted_passages"]
    assert [p["id"] for p in context] == [f"KB{i}" for i in range(1, 9)]


@pytest.mark.parametrize("value", ["0", "-1", "101", "2.5", "invalid"])
def test_invalid_retrieval_limit_rejected_before_aws(config, tmp_path, monkeypatch, value):
    monkeypatch.setenv("BEDROCK_NUMBER_OF_RESULTS", value)
    with pytest.raises(ValueError, match="BEDROCK_NUMBER_OF_RESULTS"):
        workflow(config, tmp_path, [])


@pytest.mark.parametrize("value,expected", [("", 5), ("1", 1), ("100", 100)])
def test_retrieval_limit_defaults_and_boundaries(config, tmp_path, monkeypatch, value, expected):
    monkeypatch.setenv("BEDROCK_NUMBER_OF_RESULTS", value)
    flow, client, _ = workflow(config, tmp_path, [])
    flow.run("Question")
    assert client.retrieve.call_args.kwargs["retrievalConfiguration"]["managedSearchConfiguration"]["numberOfResults"] == expected


@pytest.mark.parametrize("hits", [[], [{"content": {"type": "IMAGE"}}]])
def test_empty_evidence_skips_generation(config, tmp_path, hits):
    flow, _, model = workflow(config, tmp_path, hits)
    assert flow.run("Who approved DEV-999?").answer == INSUFFICIENT
    assert model.calls == []


@pytest.mark.parametrize("answer", ["INSUFFICIENT_EVIDENCE"])
def test_weak_or_invalid_answer_abstains(config, tmp_path, answer):
    flow, _, _ = workflow(config, tmp_path, [hit()], answer)
    assert flow.run("Who approved DEV-999?").answer == INSUFFICIENT


def test_local_control_needs_no_aws_and_uses_same_model(config, tmp_path, monkeypatch):
    monkeypatch.setenv("BEDROCK_NUMBER_OF_RESULTS", "invalid")
    monkeypatch.delenv("AWS_REGION")
    monkeypatch.delenv("BEDROCK_KNOWLEDGE_BASE_ID")
    flow, client, model = workflow(config, tmp_path, [], "I do not know.", mode="local")
    assert flow.run("Who classifies deviations?").answer == "I do not know."
    client.retrieve.assert_not_called()
    assert model.calls[0]["tools"] == []
    assert "training knowledge" in model.calls[0]["system_prompt"]
    assert "untrusted_passages" not in json.dumps(model.calls[0]["messages"])


def test_managed_document_title(config, tmp_path):
    passage = hit()
    passage["metadata"] = {"_document_title": "Fictional managed procedure", "_source_uri": "s3://fictional/file"}
    flow, _, _ = workflow(config, tmp_path, [passage])
    assert flow.run("Who classifies deviations?").sources[0].title == "Fictional managed procedure"


@pytest.mark.parametrize("answer", [
    "Here are the steps:\n\n1. Notify the quality lead.\n2. Record the review [KB1].\n\nThis completes the process.",
    "The quality lead classifies deviations [KB1].\n\n- Minor: low impact.",
    "The quality lead classifies deviations.\n\nSource: [KB1]",
])
def test_one_valid_citation_can_support_multiple_paragraphs(config, tmp_path, answer):
    flow, _, _ = workflow(config, tmp_path, [hit()], answer)
    result = flow.run("Who classifies deviations?")
    assert result.answer.startswith(answer)
    assert not result.warnings


@pytest.mark.parametrize("answer,reason", [
    ("Unsupported assertion", "missing_citations"),
    ("Invented [KB99].", "unknown_citations"),
    ("Supported [KB1]. Invented [KB99].", "unknown_citations"),
    ("[KB1]", "missing_answer"),
])
def test_invalid_citations_report_formatting_failure_not_missing_evidence(config, tmp_path, answer, reason):
    flow, _, _ = workflow(config, tmp_path, [hit()], answer)
    events = []
    flow.on_event = events.append
    result = flow.run("Who classifies deviations?")
    assert result.answer == CITATION_FAILURE
    assert result.answer != INSUFFICIENT
    validation = next(e["payload"] for e in events if e["kind"] == "kb_validation")
    assert validation["reason"] == reason


def test_missing_configuration_and_api_errors_propagate(config, tmp_path, monkeypatch):
    flow, client, model = workflow(config, tmp_path, [])
    monkeypatch.delenv("BEDROCK_KNOWLEDGE_BASE_ID")
    with pytest.raises(ValueError, match="BEDROCK_KNOWLEDGE_BASE_ID"):
        flow.run("Question")
    monkeypatch.setenv("BEDROCK_KNOWLEDGE_BASE_ID", "FAKEKB1234")
    client.retrieve.side_effect = RuntimeError("Synthetic AccessDenied")
    with pytest.raises(RuntimeError, match="AccessDenied"):
        flow.run("Question")
    assert not model.calls


def test_context_bounded_and_unknown_citation_rejected(config, tmp_path):
    flow, _, model = workflow(config, tmp_path, [hit("fictional " * 4000)] * 5)
    result = flow.run("Who classifies deviations?")
    payload = model.calls[0]
    prompt = payload["messages"][0]["content"][0]["text"]
    assert _estimated_tokens(prompt) + _estimated_tokens(payload["system_prompt"]) <= config.context_window - config.output_tokens - 512
    assert result.warnings
    tiny = replace(config, context_window=config.output_tokens + 10)
    flow, client, _ = workflow(tiny, tmp_path, [])
    with pytest.raises(ValueError, match="CONTEXT_WINDOW"):
        flow.run("Question")
    client.retrieve.assert_not_called()


@pytest.mark.parametrize("mode,expected", [(None, "research"), ("off", "local"), ("on", "rag")])
def test_cli_routing_preserves_default(config, monkeypatch, mode, expected):
    import main
    import agent.rag as rag
    from agent.orchestrator import ResearchResult
    research, local = Mock(), Mock()
    for factory in (research, local):
        factory.return_value.run.return_value = ResearchResult("test", "answer", [], [])
    monkeypatch.setattr(main, "ResearchWorkflow", research)
    monkeypatch.setattr(rag, "LocalAnswerWorkflow", local)
    monkeypatch.setattr("sys.argv", ["main.py"] + (["--rag", mode] if mode else []) + ["Question"])
    assert main.main() == 0
    if expected == "research":
        research.assert_called_once()
        local.assert_not_called()
    else:
        research.assert_not_called()
        assert local.call_args.kwargs["mode"] == expected


def test_trace_preserves_retrieved_chunks_and_exact_model_input(config, tmp_path):
    hits = [hit("  Synthetic original passage\n" + "évidence " * 1000 + "\n ")] * 5
    flow, _, model = workflow(config, tmp_path, hits)
    events = []
    flow.on_event = events.append
    result = flow.run("Who classifies deviations?")
    retrieved = next(e["payload"] for e in events if e["kind"] == "kb_retrieval_results")
    injected = next(e["payload"] for e in events if e["kind"] == "kb_context")
    request = next(e["payload"] for e in events if e["kind"] == "model_input")
    assert retrieved["results"] == hits
    assert len(injected["passages"]) < len(hits)
    assert injected["passages"][0]["text"] != hits[0]["content"]["text"]
    assert request["messages"] == model.calls[0]["messages"]
    assert request["system_prompt"] == model.calls[0]["system_prompt"]
    assert json.loads(request["messages"][0]["content"][0]["text"])["untrusted_passages"] == injected["passages"]
    assert len(injected["sources"]) == len(result.sources)
    assert injected["sources"][0]["retrieval_rank"] == 1
    assert "".join(e["payload"] for e in events if e["kind"] == "model_text") == "The quality lead classifies deviations [KB1]."


def test_source_title_link_and_returned_page_only(config, tmp_path):
    passage = hit()
    passage["metadata"]["_excerpt_page_number"] = 4.0
    passage["location"] = {"type": "WEB", "webLocation": {"url": "https://example.org/procedure.pdf"}}
    flow, _, _ = workflow(config, tmp_path, [passage])
    result = flow.run("Who classifies deviations?")
    assert "[Fictional procedure](<https://example.org/procedure.pdf>) — p. 4" in result.answer
    assert "line" not in result.answer
    record = json.loads((Path(result.run_directory) / "evidence.json").read_text())
    assert record["chunks"][0]["page"] == 4
    flow, _, _ = workflow(config, tmp_path, [hit()])
    assert " — p. " not in flow.run("Who classifies deviations?").answer


def test_source_link_is_not_clipped_with_prompt_location(config, tmp_path):
    url = "https://example.org/" + "a" * 600 + ".pdf"
    passage = hit()
    passage["location"] = {"type": "WEB", "webLocation": {"url": url}}
    flow, _, model = workflow(config, tmp_path, [passage])
    result = flow.run("Who classifies deviations?")
    assert result.sources[0].url == url
    assert f"(<{url}>)" in result.answer
    context = json.loads(model.calls[0]["messages"][0]["content"][0]["text"])["untrusted_passages"]
    assert context[0]["location"] == url[:500]


def test_context_retains_investigation_details_from_lower_ranked_hit(config, tmp_path):
    # Synthetic reproduction of the trace: five modest passages, with key steps
    # near the end of the fourth hit rather than in its introductory text.
    hits = [hit("Recorded review. " * 77), hit("Initial review. " * 45),
            hit("Notify the observer. " * 26),
            hit("Investigation task. " * 50 + "Use root-cause analysis; propose corrective actions."),
            hit("Scope and responsibilities. " * 54)]
    for h in hits:
        h["metadata"]["title"] = "Fictional investigation and review procedure"
        h["location"]["s3Location"]["uri"] = "s3://fictional/procedures/investigation-and-review-procedure.txt"
    flow, _, model = workflow(replace(config, context_window=8192, output_tokens=3072), tmp_path, hits)
    result = flow.run("Mention the steps to perform an investigation.")
    context = json.loads(model.calls[0]["messages"][0]["content"][0]["text"])["untrusted_passages"]
    assert len(context) == 5
    assert context[3]["text"].endswith("Use root-cause analysis; propose corrective actions.")
    assert not result.warnings


@pytest.mark.parametrize("text", ["review " * 5000, "知识证据" * 5000, '\\"\n\t' * 5000])
def test_context_budget_accounts_for_unicode_and_json_escaping(config, tmp_path, text):
    flow, _, model = workflow(config, tmp_path, [hit(text)] * 5)
    flow.run("Summarise the evidence.")
    request = model.calls[0]
    prompt = request["messages"][0]["content"][0]["text"]
    assert _estimated_tokens(prompt) + _estimated_tokens(request["system_prompt"]) <= config.context_window - config.output_tokens - 512
