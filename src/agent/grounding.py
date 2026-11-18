"""Conservative, local claim checks against ORIGINAL provider excerpts.

This is an abstaining verifier, not a semantic entailment model or a fact checker.
Only complete sentence matches count as supported. Narrow numeric/polarity
conflicts count as unsupported; paraphrases and absent evidence are unverifiable.
No lexical-overlap score is allowed to certify a claim.
"""
from dataclasses import dataclass
from decimal import Decimal
import re
import unicodedata

from .evidence import CITATION_PATTERN, abstract_from_inverted_index, parse_citations
from .report_validation import is_citation_metadata

_BOUNDARY = re.compile(
    r"(?<=[.!?])(?P<citations>(?:\s*\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\])*)"
    r"\s+(?=[A-Z0-9\"“])"
)
_HEADING = re.compile(
    r"^(?:#{1,6}\s|(?:\*\*|__)?(?:Perspective\s+\d+[^.!?]*|Executive summary|"
    r"Overall synthesis|Synthesis(?: and limitations)?|Summary|Final assessment|"
    r"Evidence gaps|Limitations(?: and uncertainties)?|References)(?:\*\*|__)?\s*:?$)", re.I,
)
_SEPARATOR = re.compile(r"^\s*(?:(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})$")
_SETEXT = re.compile(r"^\s*(?:={3,}|-{3,})\s*$")
_TOKENS = re.compile(r"\d+(?:\.\d+)?%?|[^\W\d_]+|[^\w\s]", re.UNICODE)
_NUMBER = re.compile(r"^\d+(?:\.\d+)?%?$")


@dataclass(frozen=True)
class ClaimCheck:
    claim_id: str
    text: str
    citations: list[str]
    status: str
    reason: str
    evidence: list[dict]
    start: int
    end: int
    reason_code: str = ""


def _tokens(text):
    text = re.sub(r"^\s*#{1,6}\s+", "", text)
    text = CITATION_PATTERN.sub("", text)
    text = unicodedata.normalize("NFKC", text).casefold().strip(" \t\r\n.*_\"'“”‘’!?:;")
    # Keep internal punctuation and ALL operators: dropping a sign/comparator
    # would incorrectly certify claims such as >= 5 versus 5, or -5 versus 5.
    return tuple(_TOKENS.findall(text))


def _spans(text):
    """Sentence-sized claims; preserve offsets for transparent inline annotation."""
    offset = 0
    paragraph_start = None
    paragraphs = []
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        # A cited factual assertion still needs checking even when formatted as
        # a heading. Ordinary headings and citation-only footers are structure.
        cited = bool(parse_citations(line))
        setext_heading = bool(line.strip() and not cited and index + 1 < len(lines)
                              and _SETEXT.match(lines[index + 1]))
        if (not line.strip() or (_HEADING.match(line.strip()) and not cited) or _SEPARATOR.match(line)
                or _SETEXT.match(line) or setext_heading or is_citation_metadata(line)):
            if paragraph_start is not None:
                paragraphs.append((paragraph_start, offset))
                paragraph_start = None
        else:
            if re.match(r"\s*(?:[-*]|\d+\.)\s", line) and paragraph_start is not None:
                paragraphs.append((paragraph_start, offset))
                paragraph_start = None
            if paragraph_start is None:
                paragraph_start = offset
        offset += len(line)
    if paragraph_start is not None:
        paragraphs.append((paragraph_start, len(text)))
    for start, end in paragraphs:
        cursor = start
        for boundary in _BOUNDARY.finditer(text, start, end):
            yield cursor, boundary.end("citations")
            cursor = boundary.end()
        yield cursor, cursor + len(text[cursor:end].rstrip())


def _sentences(text):
    sentences = []
    start = 0
    for boundary in _BOUNDARY.finditer(text):
        sentences.append(text[start:boundary.start()].strip())
        start = boundary.end()
    sentences.append(text[start:].strip())
    return [sentence for sentence in sentences if sentence]


def _original_text(store, source_id):
    try:
        raw = store.original_result(source_id)
    except (OSError, ValueError, KeyError, TypeError):
        return ""
    source = store.get(source_id)
    if not raw or source is None:
        return ""
    if source.provider == "openalex":
        return abstract_from_inverted_index(raw.get("abstract_inverted_index"))
    # Never use a generated report, the Source.snippet fallback, title or URL.
    content = raw.get("raw_content") or raw.get("content")
    return content if isinstance(content, str) else ""


def _conflict(claim, evidence):
    if len(claim) != len(evidence):
        # Only explicit copular negation, not inferred antonyms or stemming.
        for negative, positive in ((claim, evidence), (evidence, claim)):
            positions = [i for i, token in enumerate(negative) if token == "not"]
            if len(positions) != 1:
                continue
            i = positions[0]
            if (i and negative[i - 1] in {"is", "are", "was", "were"}
                    and len(positive) >= 4 and negative[:i] + negative[i + 1:] == positive):
                return "The cited excerpt states the opposite polarity."
        return None
    changed = [(a, b) for a, b in zip(claim, evidence) if a != b]
    if (len(changed) == 1 and all(_NUMBER.fullmatch(t) for t in changed[0])
            and sum(not _NUMBER.fullmatch(t) for t in claim) >= 3):
        a, b = changed[0]
        if a.endswith("%") != b.endswith("%") or Decimal(a.rstrip("%")) == Decimal(b.rstrip("%")):
            return None  # Different formatting or units is not a proven conflict.
        return f"The cited excerpt gives {changed[0][1]}, whereas the claim gives {changed[0][0]}."
    return None


def check_claims(text, store, allowed_ids):
    evidence = {}
    for source_id in allowed_ids:
        evidence[source_id] = [(sentence, _tokens(sentence))
                               for sentence in _sentences(_original_text(store, source_id))]
    checks = []
    for start, end in _spans(text):
        claim = text[start:end].strip()
        tokens = _tokens(re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", claim))
        if not tokens:
            continue
        cited = [citation[1:-1] for citation in parse_citations(claim)]
        matched, conflicts = [], []
        for source_id in cited:
            for sentence, original in evidence.get(source_id, []):
                if len(tokens) >= 3 and tokens == original:
                    matched.append({"source_id": source_id, "excerpt": sentence})
                else:
                    reason = _conflict(tokens, original)
                    if reason:
                        conflicts.append((reason, {"source_id": source_id, "excerpt": sentence}))
        if matched and conflicts:
            status = "unverifiable"
            reason = "The cited excerpts disagree; a direct match does not resolve the conflict."
            excerpts = matched + [c[1] for c in conflicts]
            code = "conflicting_sources"
        elif conflicts:
            status, reason, excerpts = "unsupported", conflicts[0][0], [c[1] for c in conflicts]
            code = "contradiction"
        elif any(source_id not in allowed_ids for source_id in cited):
            status, excerpts = "unverifiable", matched
            reason, code = "One or more cited IDs are not in the supplied evidence.", "unknown_citation"
        elif cited and any(not evidence.get(source_id) for source_id in cited):
            status, excerpts = "unverifiable", matched
            reason, code = "Original evidence is missing for one or more citations.", "missing_evidence"
        elif matched:
            status, reason, excerpts = "supported", "Complete sentence matches original retrieved text.", matched
            code = "direct_match"
        else:
            status, excerpts = "unverifiable", []
            if not cited:
                reason, code = "No source citation on this claim.", "missing_citation"
            elif any(not evidence.get(s) for s in cited):
                reason, code = "Original evidence is missing for one or more citations.", "missing_evidence"
            else:
                reason, code = ("No direct match or narrowly detectable contradiction; semantic support needs review.",
                                "semantic_review")
        checks.append(ClaimCheck(f"C{len(checks) + 1}", claim, cited, status, reason, excerpts, start, end, code))
    return checks


def annotate_grounding(text, checks):
    """Label uncertainty in the same answer card; details are saved separately."""
    for check in reversed(checks):
        if check.status == "unsupported":
            # A known contradiction must remain attached to the actual claim,
            # with the reason and original source attribution visible.
            sources = ", ".join(dict.fromkeys(item["source_id"] for item in check.evidence))
            marker = (f" **[Unsupported by cited evidence]** ({check.claim_id}: {check.reason}"
                      f" Sources: {sources}.)")
        else:
            label = {"unknown_citation": "Unknown citation", "missing_citation": "Citation needed",
                     "missing_evidence": "Evidence unavailable", "conflicting_sources": "Conflicting sources"}.get(
                         check.reason_code)
            marker = f" **[{label}: {check.claim_id}]**" if label else ""
            if check.reason_code == "semantic_review":
                marker = f" [Review {check.claim_id}]"
        text = text[:check.end] + marker + text[check.end:]
    if checks:
        counts = {status: sum(c.status == status for c in checks)
                  for status in ("supported", "unsupported", "unverifiable")}
        text += (f"\n\n*Evidence check: {counts['supported']} direct matches, {counts['unsupported']} conflicts, "
                 f"{counts['unverifiable']} not verified. Review C# marks a cited claim needing source review, "
                 "including possible valid paraphrases; no semantic support is assumed. "
                 "Contradictions and missing citations/evidence are marked inline. "
                 "Claim-level details are in Agent Activity. These checks compare retrieved excerpts, "
                 "not independent facts.*")
    return text
