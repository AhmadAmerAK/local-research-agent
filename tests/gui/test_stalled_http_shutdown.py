"""Real local sockets stall before headers or mid-Ollama-stream; Qt must close safely."""
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Event, Thread
from time import monotonic

import pytest

from agent import tools
from agent.inference import CancellableOllamaModel, OLLAMA_HTTP_TIMEOUT
from conftest import call
from gui.runtime import ResearchWorker

REAL_SEARCHES = {"academic": tools.openalex_search_async, "general": tools.tavily_search_async}


@pytest.fixture
def stalled_server():
    entered, disconnected, release = Event(), Event(), Event()
    mode = {"partial": False}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def handle_request(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            if mode["partial"]:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Connection", "close")
                self.end_headers()
                chunk = {"model": "qwen3:4b-instruct", "created_at": "2026-11-14T00:00:00Z",
                         "message": {"role": "assistant", "content": "Interrupted model output."},
                         "done": False}
                self.wfile.write(json.dumps(chunk).encode() + b"\n")
                self.wfile.flush()
            entered.set()
            self.connection.settimeout(.05)
            while not release.is_set():
                try:
                    if not self.connection.recv(1):
                        disconnected.set()
                        break
                except TimeoutError:
                    continue
                except ConnectionResetError:
                    disconnected.set()
                    break
            self.close_connection = True

        do_GET = handle_request
        do_POST = handle_request

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", mode, entered, disconnected
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


def run_and_cancel(qtbot, workflow, entered, disconnected, ready=None):
    results, failures = [], []
    worker = ResearchWorker(lambda *callbacks: workflow)
    worker.completed.connect(results.append)
    worker.failed.connect(failures.append)
    worker.start()
    worker.submit("Research clinical methods")
    try:
        qtbot.waitUntil(entered.is_set, timeout=3000)
        if ready is not None:
            qtbot.waitUntil(ready, timeout=1000)
        started = monotonic()
        worker.stop()
        qtbot.waitUntil(lambda: not worker.isRunning(), timeout=2000)
        elapsed = monotonic() - started
        qtbot.waitUntil(disconnected.is_set, timeout=1000)
        qtbot.waitUntil(lambda: bool(results) or bool(failures))
        assert not failures
        assert results[0].cancelled
        assert workflow._closed
        assert (workflow.store.directory / "conversation.json").is_file()
        print(f"shutdown_seconds={elapsed:.3f}")
        return results[0]
    finally:
        worker.stop()
        assert worker.wait(3000)


@pytest.mark.parametrize("route", ["academic", "general"])
def test_worker_shutdown_disconnects_stalled_research_http(qtbot, make_session, monkeypatch, stalled_server, route):
    url, mode, entered, disconnected = stalled_server
    provider, tool = ("openalex", "search_papers") if route == "academic" else ("tavily", "search_web")
    workflow, _, _, _, _ = make_session([
        call("select_research_route", category=route), call("read_research_instructions"),
        call(tool, query="clinical evidence"),
    ])
    monkeypatch.setattr(tools, f"{provider.upper()}_URL", url + "/search")
    monkeypatch.setattr(tools, f"{provider}_search_async", REAL_SEARCHES[route])
    result = run_and_cancel(qtbot, workflow, entered, disconnected)
    assert not result.sources
    assert not workflow.search.provider_errors  # Cancellation is not a network failure.
    assert not (workflow.store.directory / "raw_results.jsonl").exists()
    uses = {b["toolUse"]["toolUseId"] for m in workflow.agent.messages for b in m["content"] if "toolUse" in b}
    replies = {b["toolResult"]["toolUseId"] for m in workflow.agent.messages for b in m["content"] if "toolResult" in b}
    assert uses == replies


@pytest.mark.parametrize("partial", [False, True], ids=["before-headers", "mid-stream"])
def test_worker_shutdown_disconnects_stalled_ollama_http(qtbot, make_session, monkeypatch, stalled_server, partial):
    url, mode, entered, disconnected = stalled_server
    mode["partial"] = partial
    workflow, _, _, events, _ = make_session([])
    workflow.config = replace(workflow.config, ollama_host=url)
    monkeypatch.setattr(workflow, "_model", lambda output_tokens=1024: CancellableOllamaModel(
        host=url, model_id=workflow.config.ollama_model, max_tokens=output_tokens,
        ollama_client_args={"timeout": OLLAMA_HTTP_TIMEOUT},
    ))
    result = run_and_cancel(qtbot, workflow, entered, disconnected,
                            ready=(lambda: any(e["kind"] == "model_text" for e in events)) if partial else None)
    assert "cancelled" in result.answer
    if partial:
        # Streaming text is still available in activity, not presented as a checked report.
        assert any(e["kind"] == "model_text" for e in events)


@pytest.mark.parametrize("model_stall", [False, True], ids=["provider", "ollama"])
def test_stalled_http_deadline_expires_without_a_cancel_request(
        qtbot, make_session, monkeypatch, stalled_server, model_stall):
    from agent import inference
    url, mode, entered, disconnected = stalled_server
    workflow, _, _, _, _ = make_session([
        call("select_research_route", category="academic"), call("read_research_instructions"),
        call("search_papers", query="clinical"), "Research service unavailable.",
    ])
    if model_stall:
        monkeypatch.setattr(inference, "MODEL_REQUEST_DEADLINE", .15)
        monkeypatch.setattr(workflow, "_model", lambda output_tokens=1024: CancellableOllamaModel(
            host=url, model_id="qwen3:4b-instruct", ollama_client_args={"timeout": OLLAMA_HTTP_TIMEOUT}))
    else:
        monkeypatch.setattr(tools, "PROVIDER_DEADLINE", .15)
        monkeypatch.setattr(tools, "OPENALEX_URL", url + "/works")
        monkeypatch.setattr(tools, "openalex_search_async", REAL_SEARCHES["academic"])
    worker = ResearchWorker(lambda *callbacks: workflow)
    results, failures = [], []
    worker.completed.connect(results.append)
    worker.failed.connect(failures.append)
    worker.start()
    worker.submit("Research clinical methods")
    try:
        qtbot.waitUntil(entered.is_set, timeout=3000)
        qtbot.waitUntil(lambda: bool(results) or bool(failures), timeout=1500)
        qtbot.waitUntil(disconnected.is_set, timeout=1000)
        assert worker.isRunning()  # The session survives ordinary request expiry.
        if model_stall:
            assert "TimeoutError" in failures[0]
        else:
            assert not failures and not results[0].sources
            assert "couldn't retrieve sources" in results[0].answer
    finally:
        worker.stop()
        qtbot.waitUntil(lambda: not worker.isRunning(), timeout=2000)
        assert workflow._closed
