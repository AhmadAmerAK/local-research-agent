"""Offline GUI milestone tests; no Ollama or paid provider calls required."""
from pathlib import Path
import sys
from threading import Event, get_ident
from types import SimpleNamespace

from PyQt6.QtCore import QTimer, QUrl, Qt
from PyQt6.QtGui import QDesktopServices
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gui.gui_main import MessageCard, ResearchWindow, open_source
from gui.runtime import Activity, default_workflow_factory


def result(answer="## Findings\nSupported [S1].", warnings=None, incomplete=False):
    return SimpleNamespace(
        answer=answer, route="academic", warnings=warnings or [], run_directory="/tmp/run", incomplete=incomplete,
        sources=[SimpleNamespace(source_id="S1", title="A study", url="https://example.org/study")],
    )


@pytest.fixture
def make_window(qtbot):
    windows = []
    releases = []

    def make(factory):
        window = ResearchWindow(factory)
        qtbot.addWidget(window)
        window.show()
        windows.append(window)
        return window

    make.releases = releases
    yield make
    for release in releases:
        release.set()
    for window in windows:
        if window.worker is not None:
            qtbot.waitUntil(lambda: window.worker is None, timeout=5000)


def send(qtbot, window, question="Find studies on markerless gait analysis"):
    window.input.setPlainText(question)
    qtbot.mouseClick(window.send, Qt.MouseButton.LeftButton)


def test_academic_milestone_streaming_and_responsive(qtbot, make_window):
    release = Event()
    make_window.releases.append(release)
    thread_ids = []
    main_thread = get_ident()

    def factory(status, text, activity):
        thread_ids.append(get_ident())

        class FakeWorkflow:
            def run(self, question):
                assert "gait" in question
                status("Searching OpenAlex")
                activity(Activity("model_text", "I will search the literature."))
                activity(Activity("tool_call", {"name": "search_papers", "input": {"query": "gait validation"}}))
                activity(Activity("tool_result", {"result": {"new_sources": [{"id": "S1"}]}}))
                activity(Activity("model_metadata", {"stop_reason": "end_turn", "input_tokens": 200}))
                text("Draft with an unvalidated citation [S99]")
                release.wait(5)
                return result(warnings=["Limited evidence"])

        return FakeWorkflow()

    window = make_window(factory)
    ticks = []
    timer = QTimer(window)
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start()
    send(qtbot, window)
    qtbot.waitUntil(lambda: "S99" in window.answer_card.body.toPlainText())
    qtbot.waitUntil(lambda: len(ticks) >= 3)
    assert window.worker is not None
    assert not window.send.isEnabled()
    assert thread_ids == [thread_ids[0]] and thread_ids[0] != main_thread
    log = window.activity_view.toPlainText()
    for expected in ("search_papers", "gait validation", "new_sources", "end_turn", "I will search"):
        assert expected in log
    release.set()
    qtbot.waitUntil(lambda: window.worker is None)
    assert "Supported S1" in window.answer_card.body.toPlainText()
    assert "S99" not in window.answer_card.body.toPlainText()
    assert "https://example.org/study" in window.answer_card.body.toHtml()
    assert "Limited evidence" in window.activity_view.toPlainText()
    assert len(window.findChildren(MessageCard)) == 3
    assert "Limited evidence" not in window.answer_card.body.toPlainText()
    assert window.status.text() == "Research complete"
    assert "This answer is incomplete" not in window.answer_card.body.toPlainText()


def test_blank_question_and_repeated_submission(qtbot, make_window):
    questions = []

    def factory(*callbacks):
        class FakeWorkflow:
            def run(self, question):
                questions.append(question)
                return result()
        return FakeWorkflow()

    window = make_window(factory)
    window.input.setPlainText("  \n ")
    window.submit()
    assert window.worker is None and not window.send.isEnabled()
    for question in ("First question", "Second question"):
        send(qtbot, window, question)
        qtbot.waitUntil(lambda: window.worker is None)
    assert questions == ["First question", "Second question"]
    assert len(window.findChildren(MessageCard)) == 5


@pytest.mark.parametrize("factory_error", [False, True])
def test_failure_and_retry(qtbot, make_window, factory_error):
    def factory(*callbacks):
        if factory_error:
            raise ValueError("Invalid configuration")
        class FailingWorkflow:
            def run(self, question):
                raise RuntimeError("Ollama unavailable")
        return FailingWorkflow()

    window = make_window(factory)
    send(qtbot, window)
    qtbot.waitUntil(lambda: window.worker is None)
    assert "couldn't complete the research" in window.answer_card.body.toPlainText()
    assert "ERROR" in window.activity_view.toPlainText()
    window.input.setPlainText("Retry question")
    assert window.send.isEnabled()


def test_close_during_research_is_deferred(qtbot, make_window):
    release = Event()
    make_window.releases.append(release)

    def factory(*callbacks):
        class WaitingWorkflow:
            def run(self, question):
                release.wait(5)
                return result()
        return WaitingWorkflow()

    window = make_window(factory)
    send(qtbot, window)
    window.close()
    assert window.pending_close and window.isVisible()
    release.set()
    qtbot.waitUntil(lambda: window.worker is None)
    qtbot.waitUntil(lambda: not window.isVisible())


def test_links_only_open_web_urls(monkeypatch):
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    for url in ("file:///etc/passwd", "javascript:alert(1)", "https://example.org/study"):
        open_source(QUrl(url))
    assert opened == ["https://example.org/study"]


def test_gui_adapter_observes_sdk_and_preserves_backend_callback(monkeypatch):
    from agent.orchestrator import ResearchWorkflow
    from strands.hooks import BeforeToolCallEvent, AfterToolCallEvent

    callbacks = {}
    original_events = []
    activities = []
    fake_agent = SimpleNamespace(
        callback_handler=lambda **event: original_events.append(event),
        hooks=SimpleNamespace(add_callback=lambda event, callback: callbacks.update({event: callback})),
    )
    monkeypatch.setattr(ResearchWorkflow, "_agent", lambda self, **kwargs: fake_agent)
    workflow = default_workflow_factory(lambda _: None, lambda _: None, activities.append)
    assert isinstance(workflow, ResearchWorkflow)
    assert type(workflow).run is ResearchWorkflow.run
    agent = workflow._agent(tools=[], instructions="test")
    agent.callback_handler(data="Model generated text")
    agent.callback_handler(result=SimpleNamespace(
        stop_reason="end_turn", context_size=123,
        metrics=SimpleNamespace(get_summary=lambda: {"input_tokens": 123}),
    ))
    tool = {"name": "search_papers", "input": {"query": "real query"}, "toolUseId": "one"}
    callbacks[BeforeToolCallEvent](SimpleNamespace(tool_use=tool))
    callbacks[AfterToolCallEvent](SimpleNamespace(tool_use=tool, result={"content": "real result"}, duration=0.5))
    assert len(original_events) == 2
    assert [a.kind for a in activities] == ["model_text", "model_metadata", "tool_call", "tool_result"]
    assert activities[2].payload["input"]["query"] == "real query"
    assert activities[3].payload["result"] == {"content": "real result"}


def test_widget_callbacks_run_on_main_thread(qtbot, make_window):
    main_thread = get_ident()
    observed = []

    def factory(status, text, activity):
        class FakeWorkflow:
            def run(self, question):
                status("Researching")
                text("Draft")
                activity(Activity("tool_call", {"name": "search_papers"}))
                return result()
        return FakeWorkflow()

    window = make_window(factory)
    # Signal delivery from the worker to GUI-owned slots must be queued.
    original = window.show_status
    from PyQt6.QtCore import QObject, pyqtSlot
    class Observer(QObject):
        @pyqtSlot(str)
        def receive(self, text):
            observed.append(get_ident())
            original(text)
    observer = Observer(window)
    window.show_status = observer.receive
    send(qtbot, window)
    qtbot.waitUntil(lambda: window.worker is None)
    assert observed == [main_thread]


@pytest.mark.parametrize("token_limit", [True, False])
def test_failed_stream_preserves_partial_response_in_one_card(qtbot, make_window, token_limit):
    def factory(status, text, activity):
        class FailingWorkflow:
            def run(self, question):
                text("## Findings\nThe report generated so far.")
                if token_limit:
                    from strands.types.exceptions import MaxTokensReachedException
                    raise MaxTokensReachedException("Maximum token limit")
                raise ConnectionError("Connection interrupted")
        return FailingWorkflow()
    window = make_window(factory)
    send(qtbot, window)
    qtbot.waitUntil(lambda: window.worker is None)
    response = window.answer_card.body.toPlainText()
    assert "The report generated so far" in response
    assert "has been kept" not in response
    assert "couldn't complete the research" in response
    assert "MaxTokensReachedException" not in response
    assert len(window.findChildren(MessageCard)) == 3
    qtbot.wait(100)  # A pending draft-render timer must not erase the explanation.
    assert window.answer_card.body.toPlainText() == response


def test_no_evidence_warning_does_not_create_another_response(qtbot, make_window):
    def factory(*callbacks):
        class NoEvidenceWorkflow:
            def run(self, question):
                return SimpleNamespace(
                    answer="No sources were retrieved. Check your connection and try again.",
                    route="academic", sources=[], warnings=["No evidence retrieved"],
                    run_directory=None,
                )
        return NoEvidenceWorkflow()
    window = make_window(factory)
    send(qtbot, window)
    qtbot.waitUntil(lambda: window.worker is None)
    assert len(window.findChildren(MessageCard)) == 3
    assert "No evidence retrieved" not in window.answer_card.body.toPlainText()
    assert "No evidence retrieved" in window.activity_view.toPlainText()
    assert window.status.text() == "Research unavailable — please try again"


def test_token_limit_without_report_never_claims_text_was_retained(qtbot, make_window):
    def factory(status, text, activity):
        class SearchOnlyWorkflow:
            def run(self, question):
                from strands.types.exceptions import MaxTokensReachedException
                activity(Activity("model_text", "Search-stage explanation visible only in activity."))
                raise MaxTokensReachedException("Maximum token limit")
        return SearchOnlyWorkflow()
    window = make_window(factory)
    send(qtbot, window)
    qtbot.waitUntil(lambda: window.worker is None)
    response = window.answer_card.body.toPlainText()
    assert "couldn't complete the research" in response
    assert "has been kept" not in response
    assert "Search-stage explanation" in window.activity_view.toPlainText()
    assert len(window.findChildren(MessageCard)) == 3


def test_provider_failure_and_search_token_limit_together(qtbot, make_window, monkeypatch, tmp_path):
    # Reproduce the supplied trace through the actual workflow and Qt worker:
    # provider fails, search prose appears only in activity, then search hits its limit.
    from agent import orchestrator, tools as research_tools
    from agent.config import Settings
    from agent.orchestrator import ResearchWorkflow
    from strands.types.exceptions import MaxTokensReachedException
    monkeypatch.setattr(orchestrator, "AGENT_DIR", tmp_path)
    requests = []

    def unavailable(*args, **kwargs):
        requests.append(True)
        raise research_tools.ResearchAPIError("Research service unreachable or timed out")
    monkeypatch.setattr(research_tools, "tavily_search", unavailable)

    def factory(status, text, activity):
        settings = Settings("", "fake-key", "http://localhost:11434", "qwen3:4b-instruct",
                            2, 3, 2, 3, 8192, False)
        workflow = ResearchWorkflow(settings, on_status=status, on_text=text)
        workflow.classify = lambda question: "general"

        class SearchAgent:
            messages = []
            def __call__(self, prompt):
                tool_result = self.tool(query="meaning of life philosophical perspectives")
                assert tool_result["status"] == "error"
                activity(Activity("tool_result", tool_result))
                explanation = "I cannot retrieve evidence because the service is unreachable."
                self.messages = [{"role": "assistant", "content": [{"text": explanation}]}]
                activity(Activity("model_text", explanation))
                raise MaxTokensReachedException("Maximum token limit")
            def shutdown(self):
                pass

        def create(**kwargs):
            agent = SearchAgent()
            agent.tool = kwargs["tools"][0]
            return agent
        workflow._agent = create
        return workflow

    window = make_window(factory)
    send(qtbot, window, "what is the meaning of life")
    qtbot.waitUntil(lambda: window.worker is None)
    response = window.answer_card.body.toPlainText()
    assert "couldn't retrieve sources" in response
    assert "has been kept" not in response
    assert "No evidence retrieved" not in response
    assert len(requests) == 1  # No second research round after a provider outage.
    assert len(window.findChildren(MessageCard)) == 3
    assert window.status.text() == "Research unavailable — please try again"
    assert '"status": "error"' in window.activity_view.toPlainText()
    assert "RESEARCH_COMPLETED" not in window.activity_view.toPlainText()


def test_report_continuation_diagnostics_appear_only_in_activity(qtbot, make_window):
    warning = "This report reached its length limit before finishing. The text generated so far has been kept."
    def factory(*callbacks):
        class PartialReportWorkflow:
            def run(self, question):
                return result(answer="## Findings\nRetrieved evidence [S1].", warnings=[warning], incomplete=True)
        return PartialReportWorkflow()
    window = make_window(factory)
    send(qtbot, window)
    qtbot.waitUntil(lambda: window.worker is None)
    response = window.answer_card.body.toPlainText()
    assert "Retrieved evidence" in response
    assert "This answer is incomplete" in response
    assert "length limit" not in response and "has been kept" not in response
    assert warning in window.activity_view.toPlainText()
    assert len(window.findChildren(MessageCard)) == 3
    assert window.status.text() == "Partial answer available"
