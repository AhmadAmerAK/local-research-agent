"""Creation, successive turns and shutdown belong to the same long-lived worker."""
from threading import Event, get_ident

from gui.runtime import ResearchWorker


def test_worker_reuses_one_session_and_closes_on_owner_thread(qtbot):
    main = get_ident()
    lifecycle, results = [], []
    def factory(*callbacks):
        lifecycle.append(("create", get_ident()))
        class Session:
            def run(self, question):
                lifecycle.append((question, get_ident()))
                return question
            def close(self):
                lifecycle.append(("close", get_ident()))
        return Session()
    worker = ResearchWorker(factory)
    worker.completed.connect(results.append)
    worker.start()
    try:
        worker.submit("First question")
        qtbot.waitUntil(lambda: len(results) == 1)
        assert worker.isRunning()
        worker.submit("Follow-up")
        qtbot.waitUntil(lambda: len(results) == 2)
        assert worker.isRunning()
    finally:
        worker.stop()
        qtbot.waitUntil(lambda: not worker.isRunning())
    assert results == ["First question", "Follow-up"]
    assert [item[0] for item in lifecycle] == ["create", "First question", "Follow-up", "close"]
    assert len({item[1] for item in lifecycle}) == 1
    assert lifecycle[0][1] != main


def test_shutdown_cancels_active_turn_and_waits_for_close(qtbot):
    entered, closed, returned = Event(), Event(), Event()
    class Session:
        cancel_event = None
        def run(self, question):
            entered.set()
            assert self.cancel_event.wait(3)
            returned.set()
            return "Cancelled"
        def close(self):
            assert returned.is_set()
            closed.set()
    worker = ResearchWorker(lambda *callbacks: Session())
    worker.start()
    worker.submit("Long-running research")
    qtbot.waitUntil(entered.is_set)
    worker.stop()
    qtbot.waitUntil(lambda: not worker.isRunning())
    assert closed.is_set()


def test_turn_error_does_not_destroy_session_or_worker(qtbot):
    calls, results, errors = [], [], []
    class Session:
        def run(self, question):
            if question == "fail":
                raise RuntimeError("Temporary failure")
            return "Recovered"
        def close(self):
            calls.append("closed")
    def factory(*callbacks):
        calls.append("created")
        return Session()
    worker = ResearchWorker(factory)
    worker.failed.connect(errors.append)
    worker.completed.connect(results.append)
    worker.start()
    try:
        worker.submit("fail")
        qtbot.waitUntil(lambda: bool(errors))
        worker.submit("retry")
        qtbot.waitUntil(lambda: bool(results))
    finally:
        worker.stop()
        qtbot.waitUntil(lambda: not worker.isRunning())
    assert calls == ["created", "closed"]
    assert results == ["Recovered"]
