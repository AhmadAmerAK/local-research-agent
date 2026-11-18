"""Opt-in real Ollama/provider verification through the unmodified Qt GUI.

Uses the existing .env and inference settings. Saves requests, activity, rendered
answers and screenshots under /tmp by default. Not part of the offline test suite.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from gui.gui_main import MessageCard, ResearchWindow
from gui.runtime import default_workflow_factory

QUESTIONS = ("why is there a surge in the ADHD diagnoses", "how can I manage my ADHD symptoms then")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.output or Path("/tmp") / ("adhd-gui-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    directory.mkdir(parents=True, exist_ok=True)
    state = {"requests": 0, "turn": 0, "ticks": 0}
    results, errors, turns = [], [], []

    def factory(status, text, activity):
        def observe(event):
            with (directory / "activity.jsonl").open("a") as handle:
                handle.write(json.dumps({"turn": state["turn"], "kind": event.kind,
                                         "payload": event.payload}, default=str) + "\n")
            if event.kind in {"model_started", "tool_call"}:
                payload = event.payload
                print(json.dumps({"turn": state["turn"], "event": event.kind,
                                  "call": payload.get("call"), "tool": payload.get("name")}), flush=True)
            activity(event)

        workflow = default_workflow_factory(status, text, observe)
        state["workflow"] = workflow
        original_model = workflow._model

        def observed_model(*model_args, **model_kwargs):
            model = original_model(*model_args, **model_kwargs)
            original_stream = model.stream

            async def stream(messages, tool_specs=None, system_prompt=None, **kwargs):
                state["requests"] += 1
                request = model.format_request(messages, tool_specs, system_prompt)
                (directory / f"request-{state['requests']:03d}.json").write_text(
                    json.dumps({"turn": state["turn"], "request": request}, indent=2, default=str))
                source = original_stream(messages, tool_specs, system_prompt, **kwargs)
                try:
                    async for chunk in source:
                        yield chunk
                finally:
                    await source.aclose()

            model.stream = stream
            return model

        workflow._model = observed_model
        return workflow

    app = QApplication.instance() or QApplication([])
    window = ResearchWindow(factory)
    window.show()
    heartbeat = QTimer(window)
    heartbeat.setInterval(20)
    heartbeat.timeout.connect(lambda: state.__setitem__("ticks", state["ticks"] + 1))
    heartbeat.start()
    first_worker = first_agent = None
    print(f"artifacts={directory}", flush=True)
    try:
        for index, question in enumerate(QUESTIONS, 1):
            state["turn"] = index
            ticks, started = state["ticks"], monotonic()
            result_count, error_count = len(results), len(errors)
            window.input.setPlainText(question)
            QTest.mouseClick(window.send, Qt.MouseButton.LeftButton)
            if index == 1:
                first_worker = window._session_worker
                first_worker.completed.connect(results.append)
                first_worker.failed.connect(errors.append)
            while window.worker is not None:
                if monotonic() - started > 300:
                    raise TimeoutError("GUI research turn exceeded the verification deadline")
                app.processEvents()
                QTest.qWait(20)
            app.processEvents()
            workflow = state["workflow"]
            if index == 1:
                first_agent = workflow.agent
            assert window._session_worker is first_worker and workflow.agent is first_agent
            (directory / f"turn-{index}-messages.json").write_text(json.dumps(workflow.agent.messages, indent=2, default=str))
            (directory / f"turn-{index}-rendered.txt").write_text(window.answer_card.body.toPlainText())
            window.grab().save(str(directory / f"turn-{index}.png"))
            if len(errors) > error_count:
                # Record failed turns as failures, but still send the exact
                # second question through the same application session.
                failure = {"question": question, "error": errors[-1],
                           "elapsed_seconds": round(monotonic() - started, 3),
                           "same_agent": workflow.agent is first_agent,
                           "same_worker": window._session_worker is first_worker,
                           "qt_heartbeat_ticks": state["ticks"] - ticks}
                turns.append(failure)
                print(json.dumps(failure), flush=True)
                continue
            if len(results) != result_count + 1:
                raise RuntimeError("No research result delivered to the GUI")
            result = results[-1]
            assert result.sources, "No live research evidence retrieved"
            if not result.incomplete:
                assert len(result.sources) <= workflow.config.max_sources
            if index == 2:
                assert not re.search(r"(?im)^(?:#{1,6}\s+)?(?:\*\*)?Perspective\s+[123]\b", result.answer), (
                    "Routine follow-up repeated the report template")
            assert state["ticks"] - ticks >= 5, "Qt heartbeat did not remain responsive"
            (directory / f"turn-{index}-result.json").write_text(json.dumps(asdict(result), indent=2))
            turns.append({"question": question, "elapsed_seconds": round(monotonic() - started, 3),
                          "same_agent": workflow.agent is first_agent,
                          "same_worker": window._session_worker is first_worker,
                          "follow_up": workflow._follow_up, "response_format": workflow._response_format(),
                          "incomplete": result.incomplete, "warnings": result.warnings,
                          "source_ids": [s.source_id for s in result.sources],
                          "stored_sources": len(workflow.store.sources), "status": window.status.text(),
                          "qt_heartbeat_ticks": state["ticks"] - ticks,
                          "run_directory": result.run_directory})
            print(json.dumps(turns[-1]), flush=True)
        if not errors:
            assert turns[-1]["follow_up"] and turns[-1]["response_format"] == "direct_answer"
        assert len(window.findChildren(MessageCard)) == 5
        captured = [json.loads(path.read_text()) for path in directory.glob("request-*.json")]
        follow_up_requests = [item for item in captured if item["turn"] == 2]
        assert any(QUESTIONS[0] in json.dumps(item["request"]["messages"]) for item in follow_up_requests), (
            "Original question absent from all captured follow-up requests")
        (directory / "activity-rendered.txt").write_text(window.activity_view.toPlainText())
        (directory / "summary.json").write_text(json.dumps(turns, indent=2))
    finally:
        window.close()
        deadline = monotonic() + 10
        while window._session_worker is not None and monotonic() < deadline:
            app.processEvents()
            QTest.qWait(20)
        if window._session_worker is not None:
            raise RuntimeError("GUI worker did not shut down within verification deadline")
    print("GUI verification finished; session closed safely.", flush=True)
    if errors:
        raise RuntimeError("Live model errors were recorded; see summary.json")


if __name__ == "__main__":
    main()
