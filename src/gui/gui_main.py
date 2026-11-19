"""Desktop research application. Launch: python src/gui/gui_main.py."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from uuid import uuid4

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSlot
from PyQt6.QtGui import QColor, QDesktopServices, QKeySequence, QShortcut, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QFrame, QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QTextBrowser,
    QVBoxLayout, QWidget,
)

if __package__:
    from .runtime import Activity, ResearchWorker, default_retrieval_results, default_workflow_factory
else:
    from runtime import Activity, ResearchWorker, default_retrieval_results, default_workflow_factory


STYLE = """
QWidget { background: #101722; color: #e6edf5; font-family: 'DejaVu Sans'; font-size: 13px; }
QMainWindow { background: #101722; }
QLabel#brand { font-size: 24px; font-weight: bold; }
QLabel#subtitle, QLabel#hint { color: #9caec2; }
QLabel#section { font-size: 16px; font-weight: bold; }
QLabel#badge { color: #76ddc0; background: #203b3b; border-radius: 10px; padding: 7px 12px; }
QFrame#panel { background: #151f2d; border: 1px solid #29374a; border-radius: 12px; }
QFrame#panel QLabel { background: transparent; }
QFrame#message { background: #1c293a; border: 1px solid #304259; border-radius: 10px; }
QFrame#message QLabel { background: transparent; }
QLabel#role { font-weight: bold; color: #82d9c8; }
QTextBrowser { background: transparent; border: none; padding: 4px; selection-background-color: #365778; }
QPlainTextEdit { background: #0e1621; border: 1px solid #35485e; border-radius: 8px; padding: 10px; }
QPlainTextEdit:focus { border-color: #64ceb7; }
QPushButton { background: #70ddc2; color: #0a2525; border: none; border-radius: 8px; padding: 11px 22px; font-weight: bold; }
QPushButton:disabled { background: #304254; color: #8b9caf; }
QPushButton:hover { background: #96ecd6; }
QScrollArea { border: none; background: transparent; }
QSplitter::handle { background: #101722; width: 12px; }
QProgressBar { border: none; background: #253548; height: 4px; }
QProgressBar::chunk { background: #70ddc2; }
QScrollBar:vertical { background: #151f2d; width: 9px; }
QScrollBar::handle:vertical { background: #3b5068; border-radius: 4px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


def open_source(url):
    """External content may only open ordinary web links."""
    if url.scheme().lower() in {"http", "https"}:
        QDesktopServices.openUrl(url)


class MessageCard(QFrame):
    def __init__(self, role, text="", parent=None):
        super().__init__(parent)
        self.setObjectName("message")
        layout = QVBoxLayout(self)
        label = QLabel(role)
        label.setObjectName("role")
        layout.addWidget(label)
        self.body = QTextBrowser()
        self.body.setOpenLinks(False)
        self.body.setOpenExternalLinks(False)
        self.body.anchorClicked.connect(open_source)
        self.body.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.body)
        self.body.document().documentLayout().documentSizeChanged.connect(self._resize_body)
        self.set_text(text)

    def set_text(self, text, markdown=False):
        if markdown:
            self.body.setMarkdown(text)
            self.style_links()
        else:
            self.body.setPlainText(text)
        self._resize_body()

    def style_links(self):
        # Qt's default blue links are unreadable on the dark conversation cards.
        block = self.body.document().begin()
        while block.isValid():
            fragment_iterator = block.begin()
            while not fragment_iterator.atEnd():
                fragment = fragment_iterator.fragment()
                if fragment.isValid() and fragment.charFormat().isAnchor():
                    cursor = QTextCursor(self.body.document())
                    cursor.setPosition(fragment.position())
                    cursor.setPosition(fragment.position() + fragment.length(),
                                       QTextCursor.MoveMode.KeepAnchor)
                    formatting = fragment.charFormat()
                    formatting.setForeground(QColor("#8edcfa"))
                    cursor.setCharFormat(formatting)
                fragment_iterator += 1
            block = block.next()

    def _resize_body(self, *_):
        self.body.document().setTextWidth(max(100, self.body.viewport().width()))
        self.body.setFixedHeight(max(45, int(self.body.document().size().height()) + 18))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_body()


class ResearchWindow(QMainWindow):
    def __init__(self, workflow_factory=default_workflow_factory, local_factory=None, trace_directory=None):
        super().__init__()
        self.workflow_factory = workflow_factory
        self.local_factory = local_factory
        self.worker = None
        self._session_worker = None
        self.pending_close = False
        self.draft = ""
        self.answer_card = None
        self.follow_history = True
        self.trace_started_at = datetime.now(timezone.utc).isoformat()
        directory = (Path(trace_directory) if trace_directory else
                     Path(__file__).resolve().parents[1] / "agent" / "memory" / "runs" /
                     ("gui-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]))
        self.trace_path = directory / "trace.json"
        self.trace_events = []
        self._trace_saved = False
        self._run_directories = []
        self.setWindowTitle("Local Research Agent")
        self.resize(1280, 840)
        self.setMinimumSize(800, 600)
        self.setStyleSheet(STYLE)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        brand = QLabel("Local Research Agent (RAG Enhanced)")
        brand.setObjectName("brand")
        titles.addWidget(brand)
        subtitle = QLabel("Ask a question. Follow the evidence.")
        subtitle.setObjectName("subtitle")
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch()
        badge = QLabel("STRANDS HARNESS  /  OLLAMA")
        badge.setObjectName("badge")
        header.addWidget(badge)
        layout.addLayout(header)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)
        left = QFrame()
        left.setObjectName("panel")
        conversation = QVBoxLayout(left)
        conversation.setContentsMargins(16, 16, 16, 16)
        title = QLabel("Conversation")
        title.setObjectName("section")
        conversation.addWidget(title)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        history = QWidget()
        self.history = QVBoxLayout(history)
        self.history.setContentsMargins(0, 4, 6, 4)
        self.history.setSpacing(14)
        self.history.addStretch()
        self.scroll.setWidget(history)
        conversation.addWidget(self.scroll, 1)
        self.scroll.verticalScrollBar().rangeChanged.connect(self._scroll_bottom)
        self.input = QPlainTextEdit()
        self.input.setObjectName("questionInput")
        self.input.setPlaceholderText("What would you like to research?")
        self.input.setAccessibleName("Research question")
        self.input.setFixedHeight(100)
        conversation.addWidget(self.input)
        modes = QHBoxLayout()
        self.rag_toggle = QCheckBox("Use knowledge base (RAG)")
        self.reranking_toggle = QCheckBox("Managed reranking")
        self.local_toggle = QCheckBox("Local only (no web / KB)")
        self.reranking_toggle.setEnabled(False)
        for toggle in (self.rag_toggle, self.reranking_toggle, self.local_toggle):
            modes.addWidget(toggle)
        conversation.addLayout(modes)
        retrieval = QHBoxLayout()
        self.retrieval_results = QSpinBox()
        self.retrieval_results.setObjectName("retrievalResults")
        self.retrieval_results.setRange(1, 100)
        self.retrieval_results.setValue(default_retrieval_results())
        self.retrieval_results.setAccessibleName("Knowledge base results to retrieve")
        self.retrieval_results.setToolTip("Number of results for the next knowledge-base question (1–100).")
        self.retrieval_results.setEnabled(False)
        retrieval_label = QLabel("Results to retrieve:")
        retrieval_label.setBuddy(self.retrieval_results)
        retrieval.addWidget(retrieval_label)
        retrieval.addWidget(self.retrieval_results)
        retrieval.addStretch()
        conversation.addLayout(retrieval)
        self.rag_toggle.toggled.connect(self._rag_changed)
        self.local_toggle.toggled.connect(self._local_changed)
        controls = QHBoxLayout()
        hint = self.mode_hint = QLabel("Ctrl+Enter to send · Academic & web research")
        hint.setObjectName("hint")
        controls.addWidget(hint)
        controls.addStretch()
        self.send = QPushButton("Send")
        self.send.setObjectName("sendButton")
        self.send.clicked.connect(self.submit)
        controls.addWidget(self.send)
        conversation.addLayout(controls)
        splitter.addWidget(left)
        right = QFrame()
        right.setObjectName("panel")
        activity_layout = QVBoxLayout(right)
        activity_layout.setContentsMargins(16, 16, 16, 16)
        title = QLabel("Agent Activity")
        title.setObjectName("section")
        activity_layout.addWidget(title)
        description = QLabel("Live model messages, tool calls & results")
        description.setObjectName("subtitle")
        activity_layout.addWidget(description)
        self.activity_view = QPlainTextEdit()
        self.activity_view.setObjectName("activityLog")
        self.activity_view.setReadOnly(True)
        self.activity_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        activity_layout.addWidget(self.activity_view, 1)
        self.status = QLabel("Ready for a research question")
        self.status.setWordWrap(True)
        activity_layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        activity_layout.addWidget(self.progress)
        splitter.addWidget(right)
        splitter.setSizes([780, 440])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.shortcut.activated.connect(self.submit)
        self.input.textChanged.connect(self._update_send)
        self._update_send()
        self.add_message("Research assistant", "Ask an academic or general research question, or connect your Knowledge Base and chat with your documents. "
                         "Your report and sources will appear here; execution activity appears alongside.")
        self.input.setFocus()
        # Batch draft rendering to keep token streaming inexpensive.
        self.draft_timer = QTimer(self)
        self.draft_timer.setInterval(80)
        self.draft_timer.timeout.connect(self._render_draft)
        self.log_activity(Activity("session_trace", {"path": str(self.trace_path)}))

    def _scroll_bottom(self, *_):
        if self.follow_history:
            bar = self.scroll.verticalScrollBar()
            bar.setValue(bar.maximum())

    def add_message(self, role, text=""):
        card = MessageCard(role, text)
        self.history.insertWidget(self.history.count() - 1, card)
        return card

    def _update_send(self):
        self.send.setEnabled(not self.pending_close and self.worker is None and bool(self.input.toPlainText().strip()))
        idle = not self.pending_close and self.worker is None
        self.rag_toggle.setEnabled(idle)
        self.local_toggle.setEnabled(idle)
        self.reranking_toggle.setEnabled(idle and self.rag_toggle.isChecked())
        self.retrieval_results.setEnabled(idle and self.rag_toggle.isChecked())

    def _rag_changed(self, enabled):
        if enabled:
            self.local_toggle.setChecked(False)
        self._update_send()
        self._mode_hint()

    def _local_changed(self, enabled):
        if enabled:
            self.rag_toggle.setChecked(False)
        self._mode_hint()

    def _mode_hint(self):
        mode = ("Knowledge base only · no web" if self.rag_toggle.isChecked() else
                "Local model only · no web / KB" if self.local_toggle.isChecked() else
                "Academic & web research")
        self.mode_hint.setText("Ctrl+Enter to send · " + mode)

    @pyqtSlot()
    def submit(self):
        question = self.input.toPlainText().strip()
        if not question or self.worker is not None:
            return
        self.follow_history = True
        self.add_message("You", question)
        self.answer_card = self.add_message("Research assistant · working", "Starting research…")
        self.input.clear()
        self.draft = ""
        self.progress.setRange(0, 0)
        self.status.setText("Starting research…")
        self.log_activity(Activity("research_started", question))
        if self._session_worker is None:
            self._session_worker = ResearchWorker(self.workflow_factory, self, local_factory=self.local_factory)
            self._session_worker.status.connect(self.show_status)
            self._session_worker.text.connect(self.append_text)
            self._session_worker.activity.connect(self.log_activity)
            self._session_worker.completed.connect(self.show_result)
            self._session_worker.failed.connect(self.show_error)
            self._session_worker.turn_finished.connect(self.worker_finished)
            self._session_worker.finished.connect(self.session_finished)
            self._session_worker.start()
        self.worker = self._session_worker
        self._update_send()
        mode = "rag" if self.rag_toggle.isChecked() else "local" if self.local_toggle.isChecked() else "research"
        reranking = "MANAGED" if mode == "rag" and self.reranking_toggle.isChecked() else "NONE"
        self.retrieval_results.interpretText()
        number_of_results = self.retrieval_results.value() if mode == "rag" else None
        self.log_activity(Activity("answer_mode", {"mode": mode, "reranking": reranking if mode == "rag" else None,
                                                  "number_of_results": number_of_results}))
        self.worker.submit(question, mode, reranking, number_of_results)

    @pyqtSlot(str)
    def show_status(self, text):
        self.status.setText(text)
        self.log_activity(Activity("status", text))

    @pyqtSlot(str)
    def append_text(self, text):
        self.draft += text
        if not self.draft_timer.isActive():
            self.draft_timer.start()

    def _render_draft(self):
        if self.answer_card is not None:
            self.answer_card.set_text(self.draft)
        self.draft_timer.stop()

    @pyqtSlot(object)
    def log_activity(self, event):
        now = datetime.now().astimezone()
        # Freeze payloads as received; later workflow mutations cannot alter history.
        self.trace_events.append({"timestamp": now.isoformat(), "kind": event.kind,
                                  "payload": json.loads(json.dumps(event.payload, ensure_ascii=False, default=str))})
        if event.kind == "draft_reset":
            self.draft_timer.stop()
            self.draft = event.payload or ""
            if self.answer_card is not None:
                self.answer_card.set_text(self.draft or "Researching…")
            return
        cursor = self.activity_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if event.kind == "model_text":
            if getattr(self, "last_activity_kind", None) != "model_text":
                cursor.insertText("\nMODEL MESSAGE\n")
            cursor.insertText(str(event.payload))
        else:
            payload = self._activity_payload(event)
            cursor.insertText(f"\n\n{now:%H:%M:%S}  {event.kind.upper()}\n{payload}\n")
        self.last_activity_kind = event.kind
        self.activity_view.setTextCursor(cursor)
        self.activity_view.ensureCursorVisible()

    @staticmethod
    def _activity_payload(event):
        payload = event.payload
        if event.kind == "kb_retrieval_results":
            header = json.dumps({k: v for k, v in payload.items() if k != "results"}, ensure_ascii=False, indent=2)
            parts = [header]
            for rank, hit in enumerate(payload["results"], 1):
                metadata = json.dumps({k: v for k, v in hit.items() if k != "content"}, ensure_ascii=False, indent=2, default=str)
                content = hit.get("content", {})
                text = content.get("text")
                parts.append(f"RETRIEVED CHUNK {rank}\n{metadata}\nExact returned text:\n" +
                             (text if isinstance(text, str) else json.dumps(content, ensure_ascii=False)))
            return "\n\n".join(parts)
        if event.kind == "kb_context":
            header = json.dumps({k: v for k, v in payload.items() if k != "passages"}, ensure_ascii=False, indent=2, default=str)
            return header + "\n\n" + "\n\n".join(
                f"INJECTED [{p['id']}]\n{p['title']}\n{p['location']}\nExact injected excerpt:\n{p['text']}"
                for p in payload["passages"])
        return payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    @pyqtSlot(object)
    def show_result(self, result):
        self.draft_timer.stop()
        # Final validated output always replaces the unvalidated streaming draft.
        answer = result.answer
        urls = {s.source_id: s.url for s in result.sources
                if QUrl(s.url).scheme().lower() in {"http", "https"}}
        answer = re.sub(r"\[((?:S|KB)\d+)\](?!\()", lambda m: (
            f"[{m[1]}](<{urls[m[1]]}>)" if m[1] in urls else m[0]), answer)
        for warning in result.warnings:
            # Execution diagnostics belong in the activity trace. The report
            # itself supplies evidence limitations and the research outcome.
            self.log_activity(Activity("warning", warning))
        incomplete = getattr(result, "incomplete", False)
        cancelled = getattr(result, "cancelled", False)
        if incomplete and not cancelled:
            answer += "\n\n---\n\nThis answer is incomplete. Please try again for a full response."
        self.answer_card.set_text(answer, markdown=True)
        self.answer_card.findChild(QLabel, "role").setText("Research assistant")
        if result.run_directory and result.run_directory not in self._run_directories:
            self._run_directories.append(result.run_directory)
        unavailable = result.route in {"academic", "general"} and not result.sources
        self.status.setText("Research cancelled" if cancelled else
                            "Research unavailable — please try again" if unavailable else
                            "Partial answer available" if incomplete else "Research complete")
        self.log_activity(Activity("research_cancelled" if cancelled else
                                   "research_unavailable" if unavailable else
                                   "research_incomplete" if incomplete else "research_completed", {
            "route": result.route, "sources": len(result.sources),
            "run_directory": result.run_directory,
        }))
        self.follow_history = False
        QTimer.singleShot(0, self._show_answer_start)

    def _show_answer_start(self):
        self.scroll.verticalScrollBar().setValue(self.answer_card.y())

    @pyqtSlot(str)
    def show_error(self, error):
        self.draft_timer.stop()
        token_limit = "maxtokensreachedexception" in error.lower() or "maximum token" in error.lower()
        message = "I couldn't complete the research for this question. Please try again."
        retained = self.draft + "\n\n---\n\n" if self.draft else ""
        self.answer_card.set_text(retained + message, markdown=True)
        self.answer_card.findChild(QLabel, "role").setText("Research assistant")
        self.status.setText("Response length limit reached" if token_limit else "Research failed — you can retry")
        self.log_activity(Activity("error", error))

    @pyqtSlot()
    def worker_finished(self):
        self.worker = None
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self._update_send()

    @pyqtSlot()
    def session_finished(self):
        worker = self._session_worker
        self.worker = None
        self._session_worker = None
        worker.deleteLater()
        if self.pending_close:
            self.close()

    def closeEvent(self, event):
        if self._session_worker is not None:
            if not self.pending_close:
                self.log_activity(Activity("session_closing", "Waiting for workflow shutdown before saving trace."))
            self.pending_close = True
            self.status.setText("Closing research session…")
            self.input.setEnabled(False)
            self.send.setEnabled(False)
            self._session_worker.stop()
            event.ignore()
        else:
            try:
                self._save_trace()
            except (OSError, ValueError, TypeError) as exc:
                self.status.setText("Trace could not be saved. Close again to retry.")
                self.log_activity(Activity("error", f"Trace save failed: {exc}"))
                event.ignore()
                return
            event.accept()

    def _save_trace(self):
        if self._trace_saved:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.trace_path.with_suffix(".tmp")
        record = {"started_at": self.trace_started_at, "closed_at": datetime.now(timezone.utc).isoformat(),
                  "run_directories": self._run_directories, "events": self.trace_events,
                  "rendered_text": self.activity_view.toPlainText()}
        with temporary.open("w", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            json.dump(record, handle, ensure_ascii=False, indent=2)
        temporary.replace(self.trace_path)
        self._trace_saved = True
        print(f"[trace saved] {self.trace_path}", file=sys.stderr)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Local Research")
    window = ResearchWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
