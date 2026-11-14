"""Real Strands loops with scripted model streams and offline research providers."""
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from threading import get_ident

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent import orchestrator, tools
from agent.config import Settings
from agent.orchestrator import ResearchWorkflow
from strands.models.model import Model


REPORT = (
    "## Perspective 1\nClinical evidence [S1].\n"
    "## Perspective 2\nPractical evidence [S2].\n"
    "## Perspective 3\nLimitations supported by [S3]."
)


def call(name, **arguments):
    return {"tool": name, "arguments": arguments}


class ScriptedModel(Model):
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []
        self.config = {"context_window_limit": 8192}

    def update_config(self, **kwargs):
        self.config.update(kwargs)

    def get_config(self):
        return self.config

    async def structured_output(self, *args, **kwargs):
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        request = {"messages": deepcopy(messages), "tools": [tool["name"] for tool in tool_specs or []],
                   "system_prompt": system_prompt, "thread_id": get_ident()}
        self.calls.append(request)
        if not self.steps:
            raise AssertionError("Unexpected model call")
        step = self.steps.pop(0)
        if callable(step):
            step = step(request)
        if isinstance(step, Exception):
            raise step
        yield {"messageStart": {"role": "assistant"}}
        blocks = step if isinstance(step, list) else [step]
        for index, block in enumerate(blocks):
            if isinstance(block, dict) and "tool" in block:
                yield {"contentBlockStart": {"start": {"toolUse": {
                    "toolUseId": f"call-{len(self.calls)}-{index}", "name": block["tool"]}}}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {
                    "input": __import__('json').dumps(block.get("arguments", {}))}}}}
            else:
                text = block.get("text", "") if isinstance(block, dict) else block
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": text}}}
            yield {"contentBlockStop": {}}
        reason = "tool_use" if any(isinstance(block, dict) and "tool" in block for block in blocks) else "end_turn"
        if isinstance(step, dict) and step.get("limit"):
            reason = "max_tokens"
        yield {"messageStop": {"stopReason": reason}}
        yield {"metadata": {"usage": {"inputTokens": 200, "outputTokens": 30, "totalTokens": 230},
                            "metrics": {"latencyMs": 1}}}


@pytest.fixture
def make_session(monkeypatch, tmp_path):
    guide_dir = Path(orchestrator.AGENT_DIR) / "guide"
    monkeypatch.setattr(orchestrator, "AGENT_DIR", tmp_path)
    monkeypatch.setattr(orchestrator, "_read_guide", lambda: (guide_dir / "AGENTS.md").read_text())
    provider_calls = []
    sessions = []

    def papers(query, key, count=5, client=None):
        provider_calls.append(("openalex", query))
        return [{"id": f"https://openalex.org/W{i}", "title": f"Clinical study {i}",
                 "abstract_inverted_index": {"Original": [0], "evidence": [1]}} for i in range(1, 4)]

    def web(query, key, count=5, client=None):
        provider_calls.append(("tavily", query))
        return [{"url": f"https://example.org/{i}", "title": f"Web source {i}",
                 "content": "Original web evidence"} for i in range(1, 4)]

    monkeypatch.setattr(tools, "openalex_search", papers)
    monkeypatch.setattr(tools, "tavily_search", web)

    def make(steps, *, skills=False, searches=2, rounds=2):
        if skills and not (tmp_path / "guide").exists():
            shutil.copytree(guide_dir, tmp_path / "guide")
        settings = Settings("", "fake-key", "http://localhost:11434", "qwen3:4b-instruct",
                            searches, 10, rounds, 3, 8192, skills)
        model = ScriptedModel(steps)
        events, text, statuses = [], [], []
        workflow = ResearchWorkflow(settings, on_status=statuses.append, on_text=text.append, on_event=events.append)
        monkeypatch.setattr(workflow, "_model", lambda output_tokens=1024: model)
        sessions.append(workflow)
        return workflow, model, provider_calls, events, statuses

    yield make
    for workflow in sessions:
        workflow.close()
