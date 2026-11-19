"""The visible trace and ordered structured events survive normal/deferred close."""
import json
from threading import Event
from unittest.mock import Mock

from agent.config import Settings
from agent.rag import LocalAnswerWorkflow
from conftest import ScriptedModel
from gui.gui_main import ResearchWindow
from gui.runtime import Activity


def test_rag_panel_and_complete_session_export(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "synthetic-openalex")
    monkeypatch.setenv("TAVILY_API_KEY", "synthetic-tavily")
    monkeypatch.setenv("AWS_REGION", "eu-west-2")
    monkeypatch.setenv("BEDROCK_KNOWLEDGE_BASE_ID", "FAKEKB1234")
    client = Mock()
    text = "Synthetic procedure:\nThe quality lead classifies deviations.\n"
    client.retrieve.return_value = {"retrievalResults": [{
        "content": {"text": text}, "metadata": {"_document_title": "Fictional SOP", "_excerpt_page_number": 4.0},
        "location": {"type": "WEB", "webLocation": {"url": "https://example.org/procedure.pdf"}}}]}

    def factory(mode, reranking, status):
        return LocalAnswerWorkflow(Settings.from_env(), mode=mode, reranking=reranking,
                                   client=client, model=ScriptedModel(["The quality lead classifies deviations [KB1]."]),
                                   output_dir=tmp_path / "answers", on_status=status)

    window = ResearchWindow(local_factory=factory, trace_directory=tmp_path / "gui-session")
    window.resize(1600, 1000)
    window.show()
    try:
        window.rag_toggle.setChecked(True)
        window.input.setPlainText("Who classifies deviations?")
        window.submit()
        qtbot.waitUntil(lambda: window.worker is None, timeout=5000)
        panel = window.activity_view.toPlainText()
        assert "KB_RETRIEVAL_RESULTS" in panel and text in panel
        assert "KB_CONTEXT" in panel and "MODEL_INPUT" in panel
        assert "p. 4" in window.answer_card.body.toPlainText()
        assert "href=\"https://example.org/procedure.pdf\"" in window.answer_card.body.toHtml()
        for heading, filename in (("RETRIEVED CHUNK 1", "/tmp/rag-retrieved-trace.png"),
                                  ("INJECTED [KB1]", "/tmp/rag-injected-trace.png")):
            cursor = window.activity_view.document().find(heading)
            window.activity_view.setTextCursor(cursor)
            window.activity_view.centerCursor()
            qtbot.wait(20)
            window.grab().save(filename)
        window.close()
        qtbot.waitUntil(lambda: window._session_worker is None, timeout=5000)
        qtbot.waitUntil(lambda: window.trace_path.exists())
        trace = json.loads(window.trace_path.read_text())
        assert trace["rendered_text"] == window.activity_view.toPlainText()
        assert trace["run_directories"]
        retrieved = next(e for e in trace["events"] if e["kind"] == "kb_retrieval_results")
        assert retrieved["payload"]["results"][0]["content"]["text"] == text
        assert all("timestamp" in e for e in trace["events"])
        assert window.trace_path.stat().st_mode & 0o777 == 0o600
        assert not any(k in window.trace_path.read_text() for k in ("AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"))
    finally:
        window.close()
        qtbot.waitUntil(lambda: window._session_worker is None, timeout=5000)


def test_shutdown_event_and_error_are_saved_after_worker_close(qtbot, tmp_path):
    entered = Event()

    def factory(status, text, activity):
        class Workflow:
            cancel_event = None
            def run(self, question):
                entered.set()
                assert self.cancel_event.wait(3)
                raise RuntimeError("Synthetic cancelled turn")
            def close(self):
                activity(Activity("shutdown_complete", "Synthetic final event"))
        return Workflow()

    window = ResearchWindow(factory, trace_directory=tmp_path / "gui-session")
    window.show()
    window.input.setPlainText("A long synthetic question")
    window.submit()
    qtbot.waitUntil(entered.is_set)
    window.close()
    qtbot.waitUntil(lambda: window._session_worker is None, timeout=5000)
    qtbot.waitUntil(lambda: window.trace_path.exists())
    trace = json.loads(window.trace_path.read_text())
    assert "Synthetic cancelled turn" in trace["rendered_text"]
    assert "Synthetic final event" in trace["rendered_text"]
    assert any(e["kind"] == "shutdown_complete" for e in trace["events"])


def test_trace_export_failure_keeps_window_open_for_retry(qtbot, tmp_path, monkeypatch):
    window = ResearchWindow(trace_directory=tmp_path / "gui-session")
    window.show()
    original = window._save_trace
    monkeypatch.setattr(window, "_save_trace", Mock(side_effect=OSError("Synthetic disk error")))
    window.close()
    assert window.isVisible()
    assert "trace" in window.status.text().lower()
    assert "Synthetic disk error" in window.activity_view.toPlainText()
    monkeypatch.setattr(window, "_save_trace", original)
    window.close()
    assert window.trace_path.exists()
