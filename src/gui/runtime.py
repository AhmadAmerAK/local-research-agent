"""GUI boundary: injectable execution and observation of the existing workflow."""
from dataclasses import dataclass
from pathlib import Path
import sys
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
    # Load configuration before importing Strands (including telemetry setup).
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)
    source_path = str(root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from agent.config import Settings
    from agent.orchestrator import ResearchWorkflow
    from strands.hooks import BeforeToolCallEvent, AfterToolCallEvent

    class ObservedWorkflow(ResearchWorkflow):
        """Observe SDK events without replacing the backend's research logic."""

        def _agent(self, **kwargs):
            agent = super()._agent(**kwargs)
            existing_callback = agent.callback_handler

            def callback(**event):
                existing_callback(**event)
                if isinstance(event.get("data"), str):
                    on_activity(Activity("model_text", event["data"]))
                result = event.get("result")
                if result is not None:
                    on_activity(Activity("model_metadata", {
                        "model": self.config.ollama_model,
                        "stop_reason": result.stop_reason,
                        "context_size": result.context_size,
                        "metrics": result.metrics.get_summary(),
                    }))

            agent.callback_handler = callback
            agent.hooks.add_callback(BeforeToolCallEvent, lambda event: on_activity(
                Activity("tool_call", event.tool_use)))
            agent.hooks.add_callback(AfterToolCallEvent, lambda event: on_activity(
                Activity("tool_result", {
                    "tool": event.tool_use,
                    "result": event.result,
                    "duration_seconds": event.duration,
                })))
            return agent

    return ObservedWorkflow(Settings.from_env(), on_status=on_status, on_text=on_text)


class ResearchWorker(QThread):
    """Only signals cross the thread boundary; this class never touches widgets."""

    status = pyqtSignal(str)
    text = pyqtSignal(str)
    activity = pyqtSignal(object)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, question: str, factory: WorkflowFactory, parent=None):
        super().__init__(parent)
        self.question = question
        self.factory = factory

    def run(self):
        try:
            workflow = self.factory(self.status.emit, self.text.emit, self.activity.emit)
            self.completed.emit(workflow.run(self.question))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
