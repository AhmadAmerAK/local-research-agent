"""Validate report structure and citations without calling a model or provider."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from agent.evidence import EvidenceStore, Source, normalize_citations, parse_citations
from agent.report_validation import prepare_report, validate_report


VALID_REPORT = (
    "## Perspective 1 — Clinical evidence\nClinical findings [S1, S2].\n"
    "## Perspective 2 — Practical use\nPractical findings [S2].\n"
    "## Perspective 3 — Limitations\nLimitations supported by [S3].\n"
)


def test_grouped_citations_are_expanded_deduplicated_and_validated(tmp_path):
    store = EvidenceStore(tmp_path)
    store.add([Source("", "tavily", str(i), f"https://example.org/{i}", "Evidence", []) for i in range(3)])
    text = "Finding [ S1, S2; S999 ]. Other evidence [S2] and [S3]."
    assert parse_citations(text) == ["[S1]", "[S2]", "[S999]", "[S3]"]
    assert store.validate_citations(text) == (
        ["[S1]", "[S2]", "[S999]", "[S3]"], ["[S999]"],
    )
    normalized = normalize_citations(text)
    assert "[S1] [S2] [S999]" in normalized
    assert normalize_citations(normalized) == normalized
    assert parse_citations("Unrelated [text] and a single [S1].") == ["[S1]"]


@pytest.mark.parametrize("heading", ["References", "## References", "**References**", "__References:__", "###### REFERENCES ###"])
def test_model_bibliography_cannot_supply_citations_or_unverified_references(heading):
    report = "No supported narrative.\n" + heading + "\n[S1] Fabricated bibliographic text."
    assert prepare_report(report) == "No supported narrative."
    assert any("found 0" in issue for issue in validate_report(report, {"S1", "S2", "S3"}))


@pytest.mark.parametrize("style", ["markdown", "bold", "plain"])
def test_three_substantive_cited_perspectives_are_accepted(style):
    report = VALID_REPORT
    if style == "bold":
        report = "\n".join(f"**{line[3:]}**" if line.startswith("## ") else line for line in report.splitlines())
    elif style == "plain":
        report = report.replace("## ", "")
    assert validate_report(report, {"S1", "S2", "S3"}) == []


@pytest.mark.parametrize(("report", "expected"), [
    ("A report with one citation [S1].", "3 distinct valid"),
    ("Three cited sources [S1, S2, S3].", "Missing Perspective 1"),
    (VALID_REPORT.replace("Practical findings [S2].", "[S2]"), "no substantive text"),
    (VALID_REPORT.replace("Practical findings [S2].", "Practical findings."), "Perspective 2 has no valid"),
    (VALID_REPORT.replace("[S1, S2]", "[S1, S2, S999]"), "[S999]"),
    (VALID_REPORT.replace("Perspective 2", "Perspective 1"), "appears more than once"),
    (VALID_REPORT.replace("[S3]", "[S4]"), "[S4]"),
])
def test_invalid_reports_are_not_treated_as_complete(report, expected):
    assert any(expected in issue for issue in validate_report(report, {"S1", "S2", "S3"}))


def test_synthesis_citations_do_not_substitute_for_perspective_citations():
    report = VALID_REPORT.replace("Limitations supported by [S3].", "Unsupported statement.")
    report += "\n## Overall synthesis\nOther findings [S3]."
    assert "Perspective 3 has no valid supporting citation." in validate_report(report, {"S1", "S2", "S3"})


def test_insufficient_evidence_is_reported_without_demanding_invented_sources():
    issues = validate_report(VALID_REPORT.replace("S2", "S1").replace("S3", "S1"), {"S1"})
    assert len(issues) == 1 and "insufficient evidence" in issues[0]


@pytest.mark.parametrize("footer", ["Citation: [S8]", "Citations: [S8, S9]", "*Source: [S8]*",
                                    "**Sources:** [S8], [S9]", "[Citations: S8, S9]"])
def test_citation_only_footers_cannot_inflate_narrative_source_counts(footer):
    text = "An actual finding [S1].\n\n" + footer
    assert parse_citations(prepare_report(text)) == ["[S1]"]


def test_source_label_with_a_factual_assertion_is_not_discarded():
    text = "Source: The trial included 400 adults with stroke [S1]."
    assert prepare_report(text) == text
