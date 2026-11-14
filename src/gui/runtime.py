"""One background thread owns one research session until the application closes."""
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
import sys
from threading import Event
from typing import Callable, Protocol

from PyQt6.QtCore import QThread, pyqtSignal


@dataclass(frozen=True)
class Activity:
    kind: str
    payload: object


class Workflow(Protocol):
    def run(self, question: str): ...


WorkflowFactory = Callable[[Callable, Callable, Callable], Workflow]


def default_workflow_factory(on_status, on_text, on_activity):
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)
    source_path = str(root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from agent.config import Settings
    from agent.orchestrator import ResearchWorkflow
    return ResearchWorkflow(
        Settings.from_env(), on_status=on_status, on_text=on_text,
        on_event=lambda event: on_activity(Activity(event["kind"], event["payload"])),
    )


class ResearchWorker(QThread):
    """A queue serializes turns; creation, use and close all occur in run()."""
    status = pyqtSignal(str)
    text = pyqtSignal(str)
    activity = pyqtSignal(object)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    turn_finished = pyqtSignal()

    def __init__(self, factory: WorkflowFactory, parent=None):
        super().__init__(parent)
        self.factory = factory
        self._questions = Queue()
        self._stopping = Event()
        self._cancel_event = Event()

    def submit(self, question):
        if self._stopping.is_set():
            raise RuntimeError("The research session is closing.")
        self._questions.put(question)

    def stop(self):
        self._stopping.set()
        self._cancel_event.set()
        self._questions.put(None)

    def run(self):
        workflow = None
        try:
            while not self._stopping.is_set():
                question = self._questions.get()
                if question is None or self._stopping.is_set():
                    break
                try:
                    if workflow is None:
                        workflow = self.factory(self.status.emit, self.text.emit, self.activity.emit)
                        if hasattr(workflow, "cancel_event"):
                            workflow.cancel_event = self._cancel_event
                    self.completed.emit(workflow.run(question))
                except Exception as exc:
                    self.failed.emit(f"{type(exc).__name__}: {exc}")
                finally:
                    self.turn_finished.emit()
        finally:
            if workflow is not None and callable(getattr(workflow, "close", None)):
                try:
                    workflow.close()
                except Exception as exc:
                    self.activity.emit(Activity("error", f"Session shutdown: {type(exc).__name__}: {exc}"))
