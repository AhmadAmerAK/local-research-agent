"""Desktop research application. Launch: python src/gui/gui_main.py."""
from datetime import datetime
import json
import re
import sys

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSlot
from PyQt6.QtGui import QColor, QDesktopServices, QKeySequence, QShortcut, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSplitter, QTextBrowser,
    QVBoxLayout, QWidget,
)

if __package__:
    from .runtime import Activity, ResearchWorker, default_workflow_factory
else:
    from runtime import Activity, ResearchWorker, default_workflow_factory


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
    def __init__(self, workflow_factory=default_workflow_factory):
        super().__init__()
        self.workflow_factory = workflow_factory
        self.worker = None
        self.pending_close = False
        self.draft = ""
        self.answer_card = None
        self.follow_history = True
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
        brand = QLabel("Local Research Agent")
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
        controls = QHBoxLayout()
        hint = QLabel("Ctrl+Enter to send · Academic & web research")
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
        self.add_message("Research assistant", "Ask an academic or general research question. "
                         "Your report and sources will appear here; execution activity appears alongside.")
        self.input.setFocus()
        # Batch draft rendering to keep token streaming inexpensive.
        self.draft_timer = QTimer(self)
        self.draft_timer.setInterval(80)
        self.draft_timer.timeout.connect(self._render_draft)

    def _scroll_bottom(self, *_):
        if self.follow_history:
            bar = self.scroll.verticalScrollBar()
            bar.setValue(bar.maximum())

    def add_message(self, role, text=""):
        card = MessageCard(role, text)
        self.history.insertWidget(self.history.count() - 1, card)
        return card

    def _update_send(self):
        self.send.setEnabled(self.worker is None and bool(self.input.toPlainText().strip()))

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
        self.worker = ResearchWorker(question, self.workflow_factory, self)
        self.worker.status.connect(self.show_status)
        self.worker.text.connect(self.append_text)
        self.worker.activity.connect(self.log_activity)
        self.worker.completed.connect(self.show_result)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.worker_finished)
        self._update_send()
        self.worker.start()

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
        cursor = self.activity_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if event.kind == "model_text":
            if getattr(self, "last_activity_kind", None) != "model_text":
                cursor.insertText("\nMODEL MESSAGE\n")
            cursor.insertText(str(event.payload))
        else:
            payload = event.payload if isinstance(event.payload, str) else json.dumps(
                event.payload, ensure_ascii=False, indent=2, default=str)
            cursor.insertText(f"\n\n{datetime.now():%H:%M:%S}  {event.kind.upper()}\n{payload}\n")
        self.last_activity_kind = event.kind
        self.activity_view.setTextCursor(cursor)
        self.activity_view.ensureCursorVisible()

    @pyqtSlot(object)
    def show_result(self, result):
        self.draft_timer.stop()
        # Final validated output always replaces the unvalidated streaming draft.
        answer = result.answer
        urls = {s.source_id: s.url for s in result.sources
                if QUrl(s.url).scheme().lower() in {"http", "https"}}
        answer = re.sub(r"\[(S\d+)\](?!\()", lambda m: (
            f"[{m[1]}](<{urls[m[1]]}>)" if m[1] in urls else m[0]), answer)
        for warning in result.warnings:
            # Execution diagnostics belong in the activity trace. The report
            # itself supplies evidence limitations and the research outcome.
            self.log_activity(Activity("warning", warning))
        incomplete = getattr(result, "incomplete", False)
        if incomplete:
            answer += "\n\n---\n\nThis answer is incomplete. Please try again for a full response."
        self.answer_card.set_text(answer, markdown=True)
        self.answer_card.findChild(QLabel, "role").setText("Research assistant")
        unavailable = result.route in {"academic", "general"} and not result.sources
        self.status.setText("Research unavailable — please try again" if unavailable else
                            "Partial answer available" if incomplete else "Research complete")
        self.log_activity(Activity("research_unavailable" if unavailable else
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
        worker = self.worker
        self.worker = None
        worker.deleteLater()
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self._update_send()
        if self.pending_close:
            self.close()

    def closeEvent(self, event):
        if self.worker is not None:
            # The backend has no cancellation API: never destroy a running QThread.
            self.pending_close = True
            self.status.setText("Closing after the current research finishes…")
            self.input.setEnabled(False)
            event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Local Research")
    window = ResearchWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
