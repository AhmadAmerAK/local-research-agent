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


def default_retrieval_results():
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from agent.config import _integer
    count = _integer("BEDROCK_NUMBER_OF_RESULTS", 5)
    if count > 100:
        raise ValueError("BEDROCK_NUMBER_OF_RESULTS must be between 1 and 100")
    return count


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

    def __init__(self, factory: WorkflowFactory, parent=None, local_factory=None):
        super().__init__(parent)
        self.factory = factory
        self.local_factory = local_factory or local_workflow_factory
        self._questions = Queue()
        self._stopping = Event()
        self._cancel_event = Event()

    def submit(self, question, mode="research", reranking="NONE", number_of_results=None):
        if self._stopping.is_set():
            raise RuntimeError("The research session is closing.")
        if mode == "rag" and number_of_results is not None:
            if type(number_of_results) is not int or not 1 <= number_of_results <= 100:
                raise ValueError("Retrieval results must be between 1 and 100")
        self._questions.put((question, mode, reranking, number_of_results))

    def stop(self):
        self._stopping.set()
        self._cancel_event.set()
        self._questions.put(None)

    def run(self):
        workflows = {}
        try:
            while not self._stopping.is_set():
                request = self._questions.get()
                if request is None or self._stopping.is_set():
                    break
                question, mode, reranking, number_of_results = request
                try:
                    key = (mode, reranking)
                    if key not in workflows:
                        workflow = (self.factory(self.status.emit, self.text.emit, self.activity.emit)
                                    if mode == "research" else
                                    self.local_factory(mode, reranking, self.status.emit))
                        workflows[key] = workflow
                        if mode != "research" and hasattr(workflow, "on_event"):
                            workflow.on_event = lambda event: self.activity.emit(Activity(event["kind"], event["payload"]))
                        if hasattr(workflow, "cancel_event"):
                            workflow.cancel_event = self._cancel_event
                    workflow = workflows[key]
                    if mode == "rag" and number_of_results is not None:
                        workflow.number_of_results = number_of_results
                    self.completed.emit(workflow.run(question))
                except Exception as exc:
                    self.failed.emit(f"{type(exc).__name__}: {exc}")
                finally:
                    self.turn_finished.emit()
        finally:
            for workflow in workflows.values():
                try:
                    if callable(getattr(workflow, "close", None)):
                        workflow.close()
                except Exception as exc:
                    self.activity.emit(Activity("error", f"Session shutdown: {type(exc).__name__}: {exc}"))


def local_workflow_factory(mode, reranking, on_status):
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from agent.config import Settings
    from agent.rag import LocalAnswerWorkflow
    return LocalAnswerWorkflow(Settings.from_env(), mode=mode, reranking=reranking, on_status=on_status)
