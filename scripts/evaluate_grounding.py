"""Reproduce the abstaining verifier's confusion matrix on labelled claim cases."""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.evidence import EvidenceStore, normalise_tavily
from agent.grounding import check_claims


def evaluate():
    cases = json.loads((ROOT / "tests/fixtures/grounding_cases.json").read_text())
    labels = ("supported", "unsupported", "unverifiable")
    matrix = {gold: {predicted: 0 for predicted in labels} for gold in labels}
    details = []
    with tempfile.TemporaryDirectory() as directory:
        for case in cases:
            store = EvidenceStore(Path(directory) / case["id"])
            raw = {"url": "https://example.org/study", "title": "Study", "content": case["evidence"]}
            store.save_raw("tavily", "evaluation", [raw])
            store.add([normalise_tavily(raw)])
            result = check_claims(case["claim"], store, {"S1"})[0]
            matrix[case["gold"]][result.status] += 1
            details.append({"id": case["id"], "gold": case["gold"], "predicted": result.status})
    predicted_support = sum(matrix[gold]["supported"] for gold in labels)
    predicted_conflict = sum(matrix[gold]["unsupported"] for gold in labels)
    return {"cases": len(cases), "confusion_matrix_gold_rows": matrix,
            "supported_precision": matrix["supported"]["supported"] / predicted_support if predicted_support else None,
            "supported_recall": matrix["supported"]["supported"] / sum(matrix["supported"].values()),
            "unsupported_precision": matrix["unsupported"]["unsupported"] / predicted_conflict if predicted_conflict else None,
            "unsupported_recall": matrix["unsupported"]["unsupported"] / sum(matrix["unsupported"].values()),
            "abstention_rate": sum(matrix[gold]["unverifiable"] for gold in labels) / len(cases),
            "cases_detail": details,
            "limits": "Small, hand-labelled excerpt-level evaluation; not proof of research accuracy."}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
