"""Cancellation must release stalled I/O, not leave executor threads running."""
import asyncio
from threading import Event
from time import monotonic

import httpx
import pytest

from agent import tools
from agent.cancellation import RequestCancelled, cancellable


def test_cancellation_interrupts_a_stalled_await_and_runs_cleanup():
    async def scenario():
        cancel, closed = Event(), Event()

        async def stalled():
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        task = asyncio.create_task(cancellable(stalled(), cancel))
        await asyncio.sleep(.02)
        started = monotonic()
        cancel.set()
        with pytest.raises(RequestCancelled):
            await task
        assert closed.is_set()
        assert monotonic() - started < .5
    asyncio.run(scenario())


def test_provider_total_deadline_includes_retries(monkeypatch):
    async def scenario():
        attempts = []

        async def stalled(request):
            attempts.append(request)
            await asyncio.Event().wait()

        monkeypatch.setattr(tools, "PROVIDER_DEADLINE", .08)
        async with httpx.AsyncClient(transport=httpx.MockTransport(stalled)) as client:
            started = monotonic()
            with pytest.raises(tools.ResearchAPIError, match="timed out"):
                await tools.openalex_search_async("gait", "", client=client)
            assert monotonic() - started < .5
        assert len(attempts) == 1
    asyncio.run(scenario())


def test_cancel_during_retry_backoff_does_not_make_another_request():
    async def scenario():
        cancel, attempted = Event(), Event()
        attempts = []

        async def busy(request):
            attempts.append(request)
            attempted.set()
            return httpx.Response(503)

        async with httpx.AsyncClient(transport=httpx.MockTransport(busy)) as client:
            task = asyncio.create_task(tools.openalex_search_async("gait", "", client=client,
                                                                  cancel_event=cancel))
            while not attempted.is_set():
                await asyncio.sleep(.01)
            cancel.set()
            with pytest.raises(RequestCancelled):
                await task
        assert len(attempts) == 1
    asyncio.run(scenario())


def test_ollama_http_timeouts_do_not_change_inference_options():
    from agent.config import Settings
    from agent.orchestrator import ResearchWorkflow
    model = ResearchWorkflow(Settings.from_env())._model(3072)
    assert model.client_args["timeout"].read == 60
    config = model.get_config()
    assert config["max_tokens"] == 3072
    assert config["temperature"] == .15
    assert config["options"] == {"num_ctx": Settings.from_env().context_window}


def test_cancelled_turn_keeps_completed_evidence_and_supports_follow_up(make_session):
    import json
    from conftest import call
    workflow, model, providers, _, statuses = make_session([
        call("select_research_route", category="academic"), call("read_research_instructions"),
        call("search_papers", query="clinical evidence"), "Interrupted evidence [S1].",
        call("select_research_route", category="academic", follow_up=True),
        call("read_evidence", source_id="S1"), "Original evidence from the study [S1].",
    ])
    workflow.on_text = lambda text: workflow.cancel_event.set() if "Interrupted" in text else None
    first = workflow.run("Research clinical methods")
    assert first.cancelled and first.incomplete
    assert "Interrupted evidence" in first.answer and "cancelled" in first.answer
    assert statuses[-1] == "Research cancelled"
    assert len(workflow.store.sources) == 3
    assert len(json.loads((workflow.store.directory / "sources.json").read_text())) == 3
    assert workflow.store.original_result("S1") is not None
    agent = workflow.agent
    workflow.on_text = lambda _: None
    workflow.cancel_event.clear()  # Caller owns the event; only clear between turns.
    follow_up = workflow.run("Explain that study")
    assert not follow_up.cancelled and not follow_up.incomplete
    assert workflow.agent is agent and follow_up.sources[0].source_id == "S1"
    assert len(providers) == 1
    assert "Research clinical methods" in str(model.calls[-1]["messages"])


@pytest.mark.parametrize("provider", ["openalex", "tavily"])
def test_async_provider_keeps_contract_and_redacts_http_errors(provider):
    async def scenario():
        def handler(request):
            assert request.headers["authorization"] == "Bearer demo-secret"
            assert request.extensions["timeout"] == {"connect": 5, "read": 10, "write": 10, "pool": 5}
            if provider == "openalex":
                assert request.method == "GET" and request.url.params["search"] == "gait"
            else:
                assert request.method == "POST" and b'"include_answer":false' in request.content
            return httpx.Response(401, text="demo-secret")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(tools.ResearchAPIError, match="401") as exc:
                await getattr(tools, f"{provider}_search_async")("gait", "demo-secret", client=client)
            assert "demo-secret" not in str(exc.value)
    asyncio.run(scenario())


@pytest.mark.parametrize("ending", ["text", "tool", "length"])
def test_ollama_adapter_preserves_sdk_requests_chunks_and_closes_client(ending):
    import json
    from agent.inference import CancellableOllamaModel, OLLAMA_HTTP_TIMEOUT
    async def scenario():
        closed = []
        class Transport(httpx.MockTransport):
            async def aclose(self):
                closed.append(True)
                await super().aclose()
        def respond(request):
            body = json.loads(request.content)
            assert body["options"] == {"num_ctx": 8192, "num_predict": 3072, "temperature": .15}
            assert body["keep_alive"] == "5m" and body["stream"]
            message = {"role": "assistant", "content": "A short result."}
            if ending == "tool":
                message["tool_calls"] = [{"function": {"name": "read_evidence", "arguments": {"source_id": "S1"}}}]
            data = {"model": "qwen3:4b-instruct", "message": message, "done": True,
                    "done_reason": "length" if ending == "length" else "stop", "eval_count": 5,
                    "prompt_eval_count": 50, "total_duration": 1000000}
            return httpx.Response(200, content=json.dumps(data).encode() + b"\n")
        model = CancellableOllamaModel(host="http://localhost:11434", model_id="qwen3:4b-instruct",
                                      options={"num_ctx": 8192}, max_tokens=3072, temperature=.15, keep_alive="5m",
                                      ollama_client_args={"transport": Transport(respond), "timeout": OLLAMA_HTTP_TIMEOUT})
        chunks = [chunk async for chunk in model.stream([{"role": "user", "content": [{"text": "Question"}]}])]
        reason = "tool_use" if ending == "tool" else "max_tokens" if ending == "length" else "end_turn"
        assert {"messageStop": {"stopReason": reason}} in chunks
        assert chunks[-1]["metadata"]["usage"] == {"inputTokens": 50, "outputTokens": 5, "totalTokens": 55}
        assert closed == [True]
        if ending == "tool":
            assert any(chunk.get("contentBlockDelta", {}).get("delta", {}).get("toolUse") for chunk in chunks)
    asyncio.run(scenario())
