"""Known claims include deceptive high-overlap examples and valid paraphrases."""
import json
from pathlib import Path

import pytest

from agent.evidence import EvidenceStore, Source, normalise_openalex, normalise_tavily
from agent.grounding import annotate_grounding, check_claims

CASES = json.loads((Path(__file__).parents[1] / "fixtures" / "grounding_cases.json").read_text())


def evidence_store(tmp_path, text, *, academic=False):
    store = EvidenceStore(tmp_path)
    raw = {"url": "https://example.org/study", "title": "Research study", "content": text}
    normalize = normalise_tavily
    provider = "tavily"
    if academic:
        provider, normalize = "openalex", normalise_openalex
        index = {}
        for position, token in enumerate(text.split()):
            index.setdefault(token, []).append(position)
        raw = {"id": "https://openalex.org/W1", "title": "Clinical study",
               "abstract_inverted_index": index}
    store.save_raw(provider, "test", [raw])
    store.add([normalize(raw)])
    return store


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_known_claims(tmp_path, case):
    store = evidence_store(tmp_path, case["evidence"])
    checks = check_claims(case["claim"], store, {"S1"})
    assert len(checks) == 1
    assert checks[0].status == case["expected"]
    if checks[0].status == "supported":
        assert checks[0].evidence[0]["source_id"] == "S1"


def test_checks_original_openalex_abstract_not_mutated_summary(tmp_path):
    store = evidence_store(tmp_path, "The trial included 40 adults with stroke.", academic=True)
    store.sources[0].snippet = "The trial included 400 adults with stroke."
    checks = check_claims("The trial included 400 adults with stroke [S1].", store, {"S1"})
    assert checks[0].status == "unsupported"
    assert "40 adults" in checks[0].evidence[0]["excerpt"]


def test_missing_original_cannot_be_replaced_with_summary_or_title(tmp_path):
    store = EvidenceStore(tmp_path)
    store.add([Source("", "tavily", "The device measured walking speed.", "https://example.org/1",
                      "The device measured walking speed.", [])])
    assert check_claims("The device measured walking speed [S1].", store, {"S1"})[0].status == "unverifiable"


def test_claim_boundaries_include_uncited_sentences_and_citation_after_period(tmp_path):
    store = evidence_store(tmp_path, "The trial included 40 adults with stroke.")
    report = "## Perspective 1\nThe trial included 40 adults with stroke. [S1] An unrelated assertion.\n"
    checks = check_claims(report, store, {"S1"})
    assert [c.status for c in checks] == ["supported", "unverifiable"]
    assert checks[0].citations == ["S1"]
    marked = annotate_grounding(report, checks)
    assert marked.count("**[Citation needed:") == 1
    assert "An unrelated assertion. **[Citation needed: C2]**" in marked


def test_match_must_be_in_the_cited_source_not_another_retrieved_source(tmp_path):
    store = evidence_store(tmp_path, "Solar output varies with local weather.")
    raw = {"url": "https://example.org/other", "title": "Other study",
           "content": "The device measured walking speed."}
    store.save_raw("tavily", "other", [raw])
    store.add([normalise_tavily(raw)])
    checks = check_claims("The device measured walking speed [S1].", store, {"S1", "S2"})
    assert checks[0].status == "unverifiable"
    assert check_claims("The device measured walking speed [S1, S2].", store, {"S1", "S2"})[0].status == "supported"


def test_original_evidence_failure_is_unverifiable_not_success(tmp_path):
    store = evidence_store(tmp_path, "The device measured walking speed.")
    (store.directory / "raw_results.jsonl").write_text("not JSON")
    assert check_claims("The device measured walking speed [S1].", store, {"S1"})[0].status == "unverifiable"


def test_disagreeing_cited_sources_are_unverifiable_even_with_a_direct_match(tmp_path):
    store = evidence_store(tmp_path, "The trial included 40 adults with stroke.")
    raw = {"url": "https://example.org/other", "title": "Other study",
           "content": "The trial included 400 adults with stroke."}
    store.save_raw("tavily", "other", [raw])
    store.add([normalise_tavily(raw)])
    check = check_claims("The trial included 40 adults with stroke [S1, S2].", store, {"S1", "S2"})[0]
    assert check.status == "unverifiable" and "disagree" in check.reason
    assert {e["source_id"] for e in check.evidence} == {"S1", "S2"}
    assert "Conflicting sources" in annotate_grounding(check.text, [check])


@pytest.mark.parametrize("operator", ["≤", "≥", "≠", "~", "-", "+"])
def test_mathematical_operators_cannot_be_dropped_when_certifying_support(tmp_path, operator):
    store = evidence_store(tmp_path, "Mean error was 5 degrees.")
    check = check_claims(f"Mean error was {operator} 5 degrees [S1].", store, {"S1"})[0]
    assert check.status != "supported"


@pytest.mark.parametrize(("evidence", "claim"), [("40", "40.0"), ("2.5", "2.50"), ("5%", "5.0%")])
def test_equivalent_numeric_formatting_is_not_reported_as_a_conflict(tmp_path, evidence, claim):
    store = evidence_store(tmp_path, f"Reported outcome was {evidence} in adults.")
    check = check_claims(f"Reported outcome was {claim} in adults [S1].", store, {"S1"})[0]
    assert check.status == "unverifiable"


def test_headings_separators_and_setext_titles_are_not_claims_but_bold_facts_are(tmp_path):
    store = evidence_store(tmp_path, "The treatment was not clinically effective.")
    report = ("## Executive Summary\n---\n**Evidence Gaps**\n***\n"
              "**Synthesis and Limitations**\nPractical observations\n---\n"
              "**The treatment was clinically effective [S1].**\n\n[Citations: S1]\n")
    checks = check_claims(report, store, {"S1"})
    assert len(checks) == 1 and checks[0].status == "unsupported"
    assert "The treatment was clinically effective" in checks[0].text


def test_semantic_abstention_is_compact_but_contradictions_and_uncited_claims_stay_visible(tmp_path):
    store = evidence_store(tmp_path, "The trial included 40 adults with stroke.")
    report = ("Forty stroke survivors participated in the trial [S1].\n\n"
              "The trial included 400 adults with stroke [S1].\n\n"
              "The intervention prevents all future symptoms.")
    checks = check_claims(report, store, {"S1"})
    marked = annotate_grounding(report, checks)
    assert [c.status for c in checks] == ["unverifiable", "unsupported", "unverifiable"]
    assert "**[Support not verified]**" not in marked
    assert "[Review C1]" in marked
    assert "Unsupported by cited evidence" in marked and "40" in marked and "400" in marked
    assert "Citation needed" in marked
    assert "Evidence check:" in marked and marked.count("Evidence check:") == 1
    assert "source review" in marked.lower()
    assert all(c.reason for c in checks)


def test_missing_original_unknown_ids_and_disagreeing_sources_are_not_hidden_in_summary(tmp_path):
    store = evidence_store(tmp_path, "The trial included 40 adults with stroke.")
    missing = "A new assertion [S999]."
    checks = check_claims(missing, store, {"S1"})
    assert "Unknown citation" in annotate_grounding(missing, checks)
    (store.directory / "raw_results.jsonl").unlink()
    text = "The trial included 40 adults with stroke [S1]."
    assert "Evidence unavailable" in annotate_grounding(text, check_claims(text, store, {"S1"}))


def test_unknown_extra_citation_cannot_hide_a_detected_contradiction(tmp_path):
    store = evidence_store(tmp_path, "The trial included 40 adults with stroke.")
    claim = "The trial included 400 adults with stroke [S1, S999]."
    checks = check_claims(claim, store, {"S1"})
    assert checks[0].status == "unsupported"
    assert "Unsupported by cited evidence" in annotate_grounding(claim, checks)
    assert "40" in checks[0].reason and "400" in checks[0].reason


@pytest.mark.parametrize("claim", [
    "The trial included 400 adults with stroke [S1].\n---\n",
    "## The trial included 400 adults with stroke [S1].\n",
])
def test_heading_formatting_cannot_hide_a_cited_factual_contradiction(tmp_path, claim):
    store = evidence_store(tmp_path, "The trial included 40 adults with stroke.")
    checks = check_claims(claim, store, {"S1"})
    assert len(checks) == 1 and checks[0].status == "unsupported"
    assert "Unsupported by cited evidence" in annotate_grounding(claim, checks)


@pytest.mark.parametrize("footer", ["Citation: [S1]", "*Source: [S1]*", "**Sources:** [S1], [S2]"])
def test_source_only_metadata_is_not_a_claim(tmp_path, footer):
    store = evidence_store(tmp_path, "The trial included 40 adults with stroke.")
    assert check_claims(footer, store, {"S1"}) == []


def test_native_report_preserves_validation_and_exposes_grounding_without_extra_model_calls(make_session, monkeypatch):
    from conftest import call
    from agent import tools
    text = "The trial included 40 adults with stroke."
    index = {}
    for position, word in enumerate(text.split()):
        index.setdefault(word, []).append(position)
    monkeypatch.setattr(tools, "openalex_search", lambda *args: [{
        "id": f"https://openalex.org/W{i}", "title": f"Study {i}", "abstract_inverted_index": index,
    } for i in range(1, 4)])
    report = ("## Perspective 1\nThe trial included 40 adults with stroke [S1].\n"
              "## Perspective 2\nThe trial included 400 adults with stroke [S2].\n"
              "## Perspective 3\nThis may apply to other populations [S3].")
    workflow, model, _, events, statuses = make_session([
        call("select_research_route", category="academic"), call("read_research_instructions"),
        call("search_papers", query="stroke study"), report,
    ])
    result = workflow.run("Research clinical methods")
    assert [c.status for c in result.grounding] == ["supported", "unsupported", "unverifiable"]
    assert result.incomplete and statuses[-1] == "Partial report available"
    assert "**[Unsupported by cited evidence]**" in result.answer
    assert "**[Support not verified]**" not in result.answer
    assert "source review" in result.answer
    saved = json.loads(Path(result.run_directory, "grounding.json").read_text())
    assert saved[1]["status"] == "unsupported" and "40 adults" in saved[1]["evidence"][0]["excerpt"]
    assert len(model.calls) == 4
    assert any(e["kind"] == "citation_grounding" for e in events)
