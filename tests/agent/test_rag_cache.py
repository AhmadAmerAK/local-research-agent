"""Synthetic cache tests: no live credentials, documents or inference."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location("verify_rag_cached", Path(__file__).resolve().parents[2] / "scripts/verify_rag_cached.py")
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


def test_capture_reuses_exact_response(tmp_path):
    live = Mock()
    live.retrieve.return_value = {"retrievalResults": [{"content": {"text": "Synthetic procedure"}}],
                                  "ResponseMetadata": {"HTTPHeaders": {"ignored": "private transport data"}}}
    recorder = cache.CaptureClient(live, tmp_path)
    request = {"knowledgeBaseId": "FAKE", "retrievalQuery": {"text": "Question"}}
    first = recorder.retrieve(**request)
    first["retrievalResults"].clear()
    second = recorder.retrieve(**request)
    live.retrieve.assert_called_once_with(**request)
    assert second["retrievalResults"] and "ResponseMetadata" not in second
    assert next(tmp_path.glob("*.json")).stat().st_mode & 0o777 == 0o600
    entry = json.loads(next(tmp_path.glob("*.json")).read_text())
    replay = cache.ReplayClient(entry)
    assert replay.retrieve(**request) == second
    with pytest.raises(AssertionError, match="differs"):
        replay.retrieve(knowledgeBaseId="DIFFERENT")


def test_failed_attempt_is_not_retried(tmp_path):
    live = Mock()
    live.retrieve.side_effect = RuntimeError("Synthetic error")
    recorder = cache.CaptureClient(live, tmp_path)
    with pytest.raises(RuntimeError, match="Synthetic"):
        recorder.retrieve(question="First")
    with pytest.raises(RuntimeError, match="refusing"):
        recorder.retrieve(question="First")
    assert live.retrieve.call_count == 1


def test_three_attempt_budget_includes_failed_calls(tmp_path):
    live = Mock()
    live.retrieve.return_value = {"retrievalResults": []}
    recorder = cache.CaptureClient(live, tmp_path)
    for question in range(3):
        recorder.retrieve(question=question)
    with pytest.raises(RuntimeError, match="budget"):
        recorder.retrieve(question=4)
    assert live.retrieve.call_count == 3


def test_replay_network_guard_blocks_aws_and_web():
    for host in ("bedrock-agent-runtime.eu-west-2.amazonaws.com", "1.1.1.1", "api.tavily.com"):
        with pytest.raises(AssertionError, match="non-local"):
            cache.blocked_network(None, (host, 443))
