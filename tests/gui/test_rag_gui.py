"""Exercise real GUI/worker/local workflow with synthetic AWS and model responses."""
from pathlib import Path
from unittest.mock import Mock

from PyQt6.QtCore import Qt
from agent.config import Settings
from agent.rag import LocalAnswerWorkflow, INSUFFICIENT
from conftest import ScriptedModel
from gui.gui_main import ResearchWindow
from agent.orchestrator import ResearchResult


def test_modes_retrieval_and_screenshots(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-west-2")
    monkeypatch.setenv("BEDROCK_KNOWLEDGE_BASE_ID", "FAKEKB1234")
    client = Mock()
    client.retrieve.return_value = {"retrievalResults": [{
        "content": {"text": "Fictional procedure: quality lead classifies deviations."},
        "metadata": {"title": "SYNTHETIC fixture — procedure"},
        "location": {"type": "S3", "s3Location": {"uri": "s3://fictional/procedure.txt"}}}]}
    modes, research_calls = [], []

    def local_factory(mode, reranking, status):
        modes.append((mode, reranking))
        return LocalAnswerWorkflow(Settings.from_env(), mode=mode, reranking=reranking,
                                   on_status=status, client=client, output_dir=tmp_path,
                                   model=ScriptedModel(["The quality lead classifies deviations [KB1]."] * 2
                                                       if mode == "rag" else ["I cannot establish the procedure from training knowledge."]))

    class Research:
        def run(self, question):
            research_calls.append(question)
            return ResearchResult("general", "Synthetic research result", [], [])

    window = ResearchWindow(lambda *args: Research(), local_factory=local_factory)
    window.show()
    artifacts = Path("/tmp/rag-gui-screenshots")
    artifacts.mkdir(exist_ok=True)
    try:
        assert not window.rag_toggle.isChecked() and not window.local_toggle.isChecked()
        assert not window.reranking_toggle.isEnabled()
        window.grab().save(str(artifacts / "01-default.png"))
        window.local_toggle.setChecked(True)
        window.rag_toggle.setChecked(True)
        assert not window.local_toggle.isChecked()
        window.reranking_toggle.setChecked(True)
        window.input.setPlainText("Who classifies a fictional deviation?")
        qtbot.mouseClick(window.send, Qt.MouseButton.LeftButton)
        assert not window.rag_toggle.isEnabled()
        qtbot.waitUntil(lambda: window.worker is None, timeout=5000)
        assert "quality lead" in window.answer_card.body.toPlainText()
        assert "KB1" in window.answer_card.body.toPlainText()
        assert not research_calls and modes == [("rag", "MANAGED")]
        window.grab().save(str(artifacts / "02-rag-mocked.png"))
        client.retrieve.return_value = {"retrievalResults": []}
        window.input.setPlainText("Who signed fictional DEV-999?")
        window.submit()
        qtbot.waitUntil(lambda: window.worker is None, timeout=5000)
        assert INSUFFICIENT in window.answer_card.body.toPlainText()
        window.grab().save(str(artifacts / "03-insufficient-mocked.png"))
        window.local_toggle.setChecked(True)
        assert not window.rag_toggle.isChecked() and not window.reranking_toggle.isEnabled()
        calls = client.retrieve.call_count
        window.input.setPlainText("Same fictional question, local baseline")
        window.submit()
        qtbot.waitUntil(lambda: window.worker is None, timeout=5000)
        assert client.retrieve.call_count == calls and modes[-1] == ("local", "NONE")
        window.grab().save(str(artifacts / "04-local-mocked.png"))
        window.local_toggle.setChecked(False)
        window.input.setPlainText("Existing research")
        window.submit()
        qtbot.waitUntil(lambda: window.worker is None, timeout=5000)
        assert research_calls == ["Existing research"]
    finally:
        window.close()
        qtbot.waitUntil(lambda: window._session_worker is None, timeout=5000)
